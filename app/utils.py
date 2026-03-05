#extract txt from pdf
#chunk text

import fitz #PYmupdf

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = []
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text.append(page_text)
    doc.close()
    return "\n\n".join(text)

#chunk text
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    if not text:
        return []
    text = text.strip()
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

