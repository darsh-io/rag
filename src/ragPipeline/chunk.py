import pypdf
from pathlib import Path


def chunk_pdf(file_path, chunk_size=1000, overlap=200):
    pdf = pypdf.PdfReader(file_path)
    

    pages = []
    for page_num, page in enumerate(pdf.pages):
        pages.append((page_num + 1, page.extract_text() or ""))
    
    full_text = ""
    char_to_page = []
    for page_num, text in pages:
        full_text += text
        char_to_page.extend([page_num] * len(text))
    
    chunks = []
    i = 0
    while i < len(full_text):
        chunk_text = full_text[i:i+chunk_size]
        chunks.append({
            "text": chunk_text,
            "source": Path(file_path).name,
            "page": char_to_page[i],
            "chunk_index": len(chunks)
        })
        i += chunk_size - overlap
    
    return chunks

if __name__ == "__main__":
    file_path = Path(__file__).parent.parent.parent / "resources" / "Attention-Is-All-You-Need.pdf"
    chunks = chunk_pdf(file_path)
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}:\n{chunk}\n")