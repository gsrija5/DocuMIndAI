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

MODEL_NAME = "gemini-3.8-flash"

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


# ---------------- LOCAL SUMMARY ----------------

def generate_free_summary(text, max_sentences=6):
    """
    Create an extractive summary using Python only.
    No Gemini generation request is made.
    """
    sentences = get_sentences(text)

    if not sentences:
        return "No readable text was found in this PDF."

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    word_frequency = Counter(get_keywords(text))
    scored_sentences = []

    for index, sentence in enumerate(sentences):
        words = get_keywords(sentence)

        if not words:
            score = 0
        else:
            score = sum(
                word_frequency[word] for word in words
            ) / len(words)

        # Slight preference for sentences near the beginning.
        score += 1 / (index + 1)

        scored_sentences.append(
            (score, index, sentence)
        )

    selected = sorted(
        scored_sentences,
        key=lambda item: item[0],
        reverse=True,
    )[:max_sentences]

    # Keep the original document order.
    selected.sort(key=lambda item: item[1])

    return " ".join(
        item[2] for item in selected
    )


# ---------------- IMPROVED LOCAL ANSWERING ----------------

def answer_question_locally(question, pages, top_k=3):
    """Find relevant sentences and show their PDF page numbers."""

    question_words = get_keywords(question)

    if not question_words:
        return (
            "Please ask a more specific question about your PDF."
        )

    question_lower = question.lower()
    candidates = []

    for page in pages:
        for sentence in get_sentences(page["text"]):
            sentence_words = get_keywords(sentence)

            if not sentence_words:
                continue

            overlap = question_words & sentence_words

            if not overlap:
                continue

            coverage = len(overlap) / len(question_words)
            precision = len(overlap) / len(sentence_words)

            score = 0.65 * coverage + 0.35 * precision

            # Small boosts for matching question intent.
            sentence_lower = sentence.lower()

            if (
                "used for" in question_lower
                and "used for" in sentence_lower
            ):
                score += 0.25

            if (
                question_lower.startswith("what is")
                and " is " in sentence_lower
            ):
                score += 0.05

            candidates.append({
                "score": score,
                "sentence": sentence,
                "page": page["page"],
                "overlap": len(overlap),
            })

    if not candidates:
        return (
            "I couldn't find a reliable answer in this PDF. "
            "Try using keywords that appear in the document."
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = candidates[0]

    if best["overlap"] == 0 or best["score"] < 0.12:
        return "I couldn't find a reliable answer in this PDF."

    selected = []
    seen_sentences = set()

    for item in candidates:
        normalized = item["sentence"].lower().strip()

        if normalized in seen_sentences:
            continue

        if item["score"] < max(0.12, best["score"] * 0.45):
            continue

        selected.append(item)
        seen_sentences.add(normalized)

        if len(selected) >= top_k:
            break

    return "\n\n".join(
        f'{item["sentence"]} (Page {item["page"]})'
        for item in selected
    )


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

    response = call_with_retries(
        lambda: client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
    )

    if not getattr(response, "text", None):
        return "The AI did not return a text answer. Please try again."

    return response.text.strip()


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
    "Upload a PDF to summarize its content, "
    "ask questions, and download your notes."
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

            # Embeddings are optional. Local features work without them.
            try:
                with st.spinner("Preparing semantic search..."):
                    chunks = split_into_chunks(
                        document_text,
                        chunk_size=700,
                        overlap=100,
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
            "Semantic search is unavailable, possibly due to API "
            "quota limits. Local summary and keyword search remain "
            "available."
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
        limits = {
            "Short": 3,
            "Medium": 6,
            "Long": 10,
        }

        with st.spinner("Creating summary locally..."):
            st.session_state.summary = generate_free_summary(
                st.session_state.document_text,
                max_sentences=limits[summary_length],
            )

            st.session_state.summary_source = (
                "Local Python extractive summary"
            )

    if st.session_state.summary:
        st.markdown("### Your Summary")
        st.write(st.session_state.summary)

        st.caption(
            f"Summary method: {st.session_state.summary_source}"
        )

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
        answer = answer_question_locally(
            question,
            st.session_state.pages,
            top_k=3,
        )

        # Only use Gemini if local search finds no useful answer
        # and semantic embeddings are available.
        if (
            "couldn't find a reliable answer" in answer.lower()
            and st.session_state.chunks
            and st.session_state.chunk_vectors
        ):
            try:
                with st.spinner("Trying AI-powered search..."):
                    relevant_chunks = retrieve_relevant_chunks(
                        question,
                        st.session_state.chunks,
                        st.session_state.chunk_vectors,
                        top_k=3,
                    )

                    context = "\n\n".join(relevant_chunks)

                    answer = answer_question_with_gemini(
                        question,
                        context,
                        st.session_state.chat_history_for_gemini,
                    )

            except Exception:
                answer = (
                    "I couldn't find a reliable answer locally, "
                    "and AI-powered answering is currently "
                    "unavailable. Try using keywords from the PDF."
                )

        st.session_state.messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])

        st.session_state.chat_history_for_gemini.extend([
            {"role": "user", "content": question},
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