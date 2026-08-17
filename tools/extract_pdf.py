import sys
from pypdf import PdfReader

try:
    reader = PdfReader('/Users/zangzelin/code/mldr_rz/报告.pdf')
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- Page {i+1} ---\n"
        text += page.extract_text() + "\n"
    
    # Print to stdout instead of file to avoid file creation permission issues if any,
    # and to keep it simple.
    print(text)
except Exception as e:
    print(f"Error extracting PDF: {e}")
