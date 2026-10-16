import pymupdf

# Open the PDF file
pdf = pymupdf.open("sample.pdf")

# Read text from every page
document_text = ""

for page in pdf:
    document_text += page.get_text() + "\n"

pdf.close()

# Display the extracted text
print("Text extracted from PDF:")
print("-------------------------")
print(document_text)