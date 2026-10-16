import hashlib
import re
from collections import Counter

import fitz
import streamlit as st

from retrieval import (
    split_into_chunks,
    create_embeddings,
    retrieve_relevant_chunks,
    call_with_retries,
    client,
)

# ---------------- CONFIGURATION ----------------

PRIMARY_MODEL = "gemini-3.8-flash"
BACKUP_MODEL = "gemini-3.5-flash-lite"
MODEL_NAMES = (PRIMARY_MODEL, BACKUP_MODEL)

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
)

# ---------------- TEXT PROCESSING ----------------

STOP_WORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and", "or",
    "to", "of", "in", "on", "for", "with", "by", "as", "at",
    "it", "this", "that", "these", "those", "be", "been", "being",
    "from", "which", "what", "when", "where", "who", "why", "how",
    "can", "could", "would", "should", "do", "does", "did",
    "about", "into", "their", "there", "they", "them", "its",
    "you", "your", "i", "we", "our", "he", "she", "not",
    "use", "used", "using",
}


def extract_pdf_pages(pdf_file):
    """Extract text from each PDF page."""
    pages = []

    with fitz.open(
        stream=pdf_file.getvalue(),
        filetype="pdf",
    ) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    return pages


def get_sentences(text):
    """Split text into sentences."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def get_keywords(text):
    """Extract meaningful words from text."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        word for word in words
        if len(word) > 1 and word not in STOP_WORDS
    }


# ---------------- LOCAL FALLBACK HELPERS ----------------

def generate_local_summary(text, max_sentences=6):
    """Create an extractive summary only if Gemini fails."""
    sentences = get_sentences(text)
    if not sentences:
        return "No readable text was found in this PDF."
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    frequencies = Counter(
        word for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(word) > 1 and word not in STOP_WORDS
    )
    scored = []
    for index, sentence in enumerate(sentences):
        words = get_keywords(sentence)
        score = (sum(frequencies[word] for word in words) / len(words)) if words else 0
        score += 1 / (index + 1)
        scored.append((score, index, sentence))
    chosen = sorted(scored, reverse=True)[:max_sentences]
    chosen.sort(key=lambda item: item[1])
    return " ".join(item[2] for item in chosen)


def answer_question_locally(question, pages, top_k=3):
    """Find matching sentences locally as a last-resort fallback."""
    question_words = get_keywords(question)
    if not question_words:
        return "Please ask a more specific question about your PDF."

    candidates = []
    for page in pages:
        for sentence in get_sentences(page["text"]):
            sentence_words = get_keywords(sentence)
            overlap = question_words & sentence_words
            if not overlap:
                continue
            coverage = len(overlap) / max(len(question_words), 1)
            precision = len(overlap) / max(len(sentence_words), 1)
            candidates.append({
                "score": 0.65 * coverage + 0.35 * precision,
                "sentence": sentence,
                "page": page["page"],
            })

    if not candidates:
        return "I couldn't find a matching answer in the PDF using local search."

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected, seen = [], set()
    best_score = candidates[0]["score"]
    for item in candidates:
        normalized = item["sentence"].lower().strip()
        if normalized in seen or item["score"] < max(0.08, best_score * 0.4):
            continue
        selected.append(item)
        seen.add(normalized)
        if len(selected) >= top_k:
            break
    return "\n\n".join(
        f'{item["sentence"]} (Page {item["page"]})' for item in selected
    )


# ---------------- GEMINI MODEL FALLBACK ----------------

def generate_with_gemini_fallback(prompt):
    """Try the primary Gemini model, then the tested backup model."""
    errors = []
    for model_name in MODEL_NAMES:
        try:
            response = call_with_retries(
                lambda model_name=model_name: client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
            )
            if not getattr(response, "text", None):
                raise RuntimeError(f"{model_name} returned an empty response.")
            return response.text.strip(), model_name
        except Exception as error:
            errors.append(f"{model_name}: {error}")
    raise RuntimeError("Both Gemini models failed. " + " | ".join(errors))


# ---------------- GEMINI SUMMARY ----------------

def generate_gemini_summary(text, summary_length):
    """Generate an abstractive summary with Gemini, not local extraction."""
    instructions = {
        "Short": "Write a concise summary in 3-5 bullet points.",
        "Medium": "Write a clear summary with headings and 6-10 key points.",
        "Long": "Write a detailed summary with headings, key concepts, explanations, and takeaways.",
    }

    prompt = f"""
You are DocuMind AI, an AI assistant that summarizes PDF documents.
Summarize only the source text below. Do not add facts from outside the document.
Use clear, simple English and preserve important terms and details.
{instructions[summary_length]}
If the document does not contain enough detail for a requested point, do not invent it.

PDF TEXT:
{text[:60000]}

AI SUMMARY:
"""

    summary, model_name = generate_with_gemini_fallback(prompt)
    return summary, model_name


# ---------------- OPTIONAL GEMINI ANSWERING ----------------

def answer_question_with_gemini(
    question,
    context,
    chat_history,
):
    """Generate an answer using retrieved PDF context."""

    history_text = ""

    for message in chat_history[-6:]:
        role = message["role"].capitalize()
        history_text += (
            f'{role}: {message["content"]}\n'
        )

    prompt = f"""
You are DocuMind AI, a document question-answering assistant.

Answer using only the PDF context below.
Give a direct, clear answer first, then explain briefly.
Include page numbers when the context provides them.
If the answer is missing from the context, say so.
Never invent facts.

Previous conversation:
{history_text}

PDF context:
{context}

Question:
{question}
"""

    answer, model_name = generate_with_gemini_fallback(prompt)
    return answer, model_name


# ---------------- SESSION STATE ----------------

defaults = {
    "pdf_hash": None,
    "pages": [],
    "chunks": [],
    "chunk_vectors": [],
    "document_text": "",
    "messages": [],
    "summary": "",
    "summary_source": "",
    "summary_warning": "",
    "embedding_error": "",
    "chat_history_for_gemini": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- APP INTERFACE ----------------

st.title("📚 DocuMind AI")
st.subheader("Your AI-powered PDF study assistant")

st.write(
    "Upload a PDF to generate Gemini summaries, ask AI questions grounded in the document, "
    "and download your notes."
)

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"],
)

# ---------------- PDF UPLOAD ----------------

if uploaded_file is not None:
    current_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    if current_hash != st.session_state.pdf_hash:
        try:
            with st.spinner("Reading your PDF..."):
                pages = extract_pdf_pages(uploaded_file)

                if not pages:
                    st.error(
                        "No readable text was found. "
                        "The PDF may contain scanned images."
                    )
                    st.stop()

                document_text = "\n\n".join(
                    page["text"] for page in pages
                )

                st.session_state.pdf_hash = current_hash
                st.session_state.pages = pages
                st.session_state.document_text = document_text
                st.session_state.chunks = []
                st.session_state.chunk_vectors = []
                st.session_state.embedding_error = ""
                st.session_state.messages = []
                st.session_state.chat_history_for_gemini = []
                st.session_state.summary = ""
                st.session_state.summary_source = ""
                st.session_state.summary_warning = ""

            # Embeddings are optional. Local features work without them.
            try:
                with st.spinner("Preparing semantic search..."):
                    # Keep page labels inside each chunk so Gemini can cite sources.
                    chunks = []
                    for page in pages:
                        page_chunks = split_into_chunks(
                            page["text"],
                            chunk_size=700,
                            overlap=100,
                        )
                        chunks.extend(
                            [f"[Page {page['page']}]\n{chunk}" for chunk in page_chunks]
                        )

                    vectors = create_embeddings(chunks)

                    st.session_state.chunks = chunks
                    st.session_state.chunk_vectors = vectors

            except Exception as error:
                st.session_state.chunks = []
                st.session_state.chunk_vectors = []
                st.session_state.embedding_error = str(error)

        except Exception as error:
            st.error(f"Could not process the PDF: {error}")
            st.stop()

    st.success("PDF loaded successfully!")

    st.caption(
        f"Pages with readable text: "
        f"{len(st.session_state.pages)}"
    )

    if st.session_state.embedding_error:
        st.info(
            "Semantic search is unavailable, possibly because of an API quota or service issue. "
            "Gemini summary and question answering can still be attempted, but Q&A may use the full PDF text."
        )

    # ---------------- SUMMARY ----------------

    st.divider()
    st.header("📝 PDF Summary")

    summary_length = st.selectbox(
        "Choose summary length",
        ["Short", "Medium", "Long"],
        index=1,
    )

    if st.button("Generate Summary", type="primary"):
        with st.spinner("Trying Gemini first..."):
            try:
                summary, model_name = generate_gemini_summary(
                    st.session_state.document_text,
                    summary_length,
                )
                st.session_state.summary = summary
                st.session_state.summary_source = f"Gemini AI ({model_name})"
                st.session_state.summary_warning = ""
            except Exception as error:
                limits = {"Short": 3, "Medium": 6, "Long": 10}
                st.session_state.summary = generate_local_summary(
                    st.session_state.document_text,
                    max_sentences=limits[summary_length],
                )
                st.session_state.summary_source = "Local Python fallback"
                st.session_state.summary_warning = (
                    "Both Gemini models were unavailable, so this summary was generated locally. "
                    f"Reason: {error}"
                )

    if st.session_state.summary:
        st.markdown("### Your Summary")
        st.write(st.session_state.summary)

        st.caption(
            f"Summary method: {st.session_state.summary_source}"
        )
        if st.session_state.get("summary_warning"):
            st.warning(st.session_state.summary_warning)

        st.download_button(
            "⬇️ Download Summary",
            data=st.session_state.summary,
            file_name="DocuMind_AI_Summary.txt",
            mime="text/plain",
        )

    # ---------------- QUESTION ANSWERING ----------------

    st.divider()
    st.header("💬 Ask Questions About Your PDF")

    st.write(
        "Ask questions about the uploaded document. "
        "Answers include page references when available."
    )

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="Example: What is Python used for?",
        )

        submitted = st.form_submit_button("Get Answer")

    if submitted and question.strip():
        user_question = question.strip()
        answer_source = ""
        gemini_error = ""

        try:
            with st.spinner("Trying Gemini first..."):
                context = ""
                if st.session_state.chunks and st.session_state.chunk_vectors:
                    try:
                        retrieved = retrieve_relevant_chunks(
                            user_question,
                            st.session_state.chunks,
                            st.session_state.chunk_vectors,
                            top_k=4,
                        )
                        # retrieval.py returns (similarity_score, index, chunk) tuples.
                        context_parts = [item[2] for item in retrieved if len(item) >= 3]
                        context = "\n\n".join(context_parts)
                    except Exception as retrieval_error:
                        st.warning(
                            "Semantic retrieval failed; trying Gemini with the "
                            "PDF's extracted text instead."
                        )
                        st.caption(str(retrieval_error))

                # If embeddings are unavailable, still try Gemini using PDF text.
                if not context.strip():
                    context = st.session_state.document_text[:40000]
                if not context.strip():
                    raise RuntimeError("No readable PDF context is available for this question.")

                answer, model_name = answer_question_with_gemini(
                    user_question,
                    context,
                    st.session_state.chat_history_for_gemini,
                )
                answer_source = f"Gemini AI ({model_name})"

        except Exception as error:
            gemini_error = str(error)
            answer = answer_question_locally(
                user_question,
                st.session_state.pages,
                top_k=3,
            )
            answer_source = "Local Python fallback"

        if gemini_error:
            st.warning(
                "Both Gemini models were unavailable, so local Python search was used as a fallback. "
                f"Reason: {gemini_error}"
            )

        displayed_answer = f"**Source: {answer_source}**\n\n{answer}"
        st.session_state.messages.extend([
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": displayed_answer},
        ])
        if answer_source.startswith("Gemini"):
            st.session_state.chat_history_for_gemini.extend([
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": answer},
            ])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history_for_gemini = []
        st.rerun()

else:
    st.info("Upload a PDF to get started.")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "DocuMind AI | Python | Streamlit | Gemini | "
    "PDF Search | Retrieval-Augmented Generation (RAG)"
)