import os
import re
from pdfminer.high_level import extract_text


def clean_text(text):
    """
    Cleans the extracted text by removing unnecessary characters and whitespace.

    Args:
        text (str): The extracted text from the PDF file.

    Returns:
        str: The cleaned text.
    """
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfminer.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF file.
    """
    raw_text = extract_text(pdf_path)
    return raw_text


# Example usage
if __name__ == "__main__":
    pdf_path = "sample.pdf"

    # Extract using pdfminer
    extracted_text = extract_text_from_pdf(pdf_path)

    # Print extracted text
    print("Extracted Text:")
    print(extracted_text)
