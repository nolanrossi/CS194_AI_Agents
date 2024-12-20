from pypdf import PdfReader

def check_pdf_type(file_path):
    reader = PdfReader(file_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:  # If text extraction works
            return "The PDF is clean/text-based."
    return "The PDF is scanned (image-based)."

pdf_file = "textbook.pdf"
result = check_pdf_type(pdf_file)
print(result)
