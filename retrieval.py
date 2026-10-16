import os
import time
import math

from dotenv import load_dotenv
from google import genai
from google.genai import types


# --------------------------------------------------
# GEMINI CONFIGURATION
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=120000),
)

EMBEDDING_MODEL = "gemini-embedding-001"


# --------------------------------------------------
# API RETRY HANDLING
# --------------------------------------------------

def call_with_retries(function, max_attempts=3):
    """Retry temporary errors without repeatedly retrying exhausted quotas."""

    for attempt in range(max_attempts):
        try:
            return function()

        except Exception as error:
            message = str(error).upper()

            quota_exhausted = (
                "QUOTA EXCEEDED" in message
                or "GENERATE_CONTENT_FREE_TIER_REQUESTS" in message
                or "EXCEEDED YOUR CURRENT QUOTA" in message
            )

            if quota_exhausted:
                raise RuntimeError(
                    "Gemini's free-tier quota is exhausted. "
                    "Wait for the quota to reset."
                ) from error

            retryable = any(
                marker in message
                for marker in (
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "502",
                    "504",
                    "INTERNAL",
                    "TIMEOUT",
                )
            )

            if not retryable or attempt == max_attempts - 1:
                raise

            # A 429 can mean a temporary rate limit or an
            # exhausted quota. Avoid repeated rapid requests.
            wait_seconds = 2 ** (attempt + 1)

            print(
                f"Temporary API error. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def split_into_chunks(text, chunk_size=700, overlap=100):
    """Split text into overlapping chunks of words."""

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk:
            chunks.append(chunk)

    return chunks


# --------------------------------------------------
# EMBEDDING CREATION
# --------------------------------------------------

def create_embeddings(texts, batch_size=50):
    """Create embeddings in batches."""

    if not texts:
        return []

    vectors = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = call_with_retries(
            lambda batch=batch: client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
            )
        )

        if not response.embeddings:
            raise RuntimeError(
                "Gemini returned no embeddings."
            )

        batch_vectors = [
            embedding.values
            for embedding in response.embeddings
        ]

        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                "Embedding count does not match text count."
            )

        vectors.extend(batch_vectors)

    return vectors


# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    """Measure similarity between two embedding vectors."""

    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# --------------------------------------------------
# RETRIEVE RELEVANT PDF CHUNKS
# --------------------------------------------------

def retrieve_relevant_chunks(
    question,
    chunks,
    chunk_vectors,
    top_k=3,
):
    """Find the PDF chunks most relevant to a question."""

    if not question or not question.strip():
        return []

    if not chunks or not chunk_vectors:
        return []

    if len(chunks) != len(chunk_vectors):
        raise ValueError(
            "Each text chunk needs a matching embedding."
        )

    if top_k <= 0:
        return []

    response = call_with_retries(
        lambda: client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question,
        )
    )

    if not response.embeddings:
        raise RuntimeError(
            "Gemini returned no embedding for the question."
        )

    question_vector = response.embeddings[0].values

    results = []

    for index, vector in enumerate(chunk_vectors):
        score = cosine_similarity(question_vector, vector)

        results.append(
            (score, index, chunks[index])
        )

    results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return results[:min(top_k, len(results))]