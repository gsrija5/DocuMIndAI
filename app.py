import pymupdf
import re

print("Welcome to DocuMind AI!")

# Step 1: Read the PDF
try:
    pdf = pymupdf.open("sample.pdf")
    document_text = ""

    for page in pdf:
        document_text += page.get_text() + "\n"

    pdf.close()

except Exception as error:
    print("Could not read the PDF:", error)
    raise SystemExit

# Step 2: Split the PDF into sentences
sentences = re.split(r"(?<=[.!?])\s+", document_text.strip())

# Step 3: Ask a question
question = input("Ask a question about the PDF: ").lower()

# Step 4: Extract useful keywords
stop_words = {
    "what", "is", "are", "the", "a", "an",
    "how", "does", "do", "in", "of", "to",
    "for", "and", "from", "this", "that",
    "can", "used", "use", "with", "it"
}

question_words = set(re.findall(r"\b[a-z0-9]+\b", question))
keywords = question_words - stop_words

# Step 5: Score each sentence
results = []

for sentence in sentences:
    sentence = sentence.strip()

    if not sentence:
        continue

    sentence_words = set(
        re.findall(r"\b[a-z0-9]+\b", sentence.lower())
    )

    score = len(keywords.intersection(sentence_words))

    if score > 0:
        results.append((score, sentence))

# Step 6: Display the best matching answers
print("\nDocuMind AI Answer:")

if results:
    results.sort(key=lambda item: item[0], reverse=True)

    best_score = results[0][0]

    for score, sentence in results:
        if score == best_score:
            print("-", sentence)
else:
    print("I couldn't find relevant information in the PDF.")