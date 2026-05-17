import pypdf
from pathlib import Path


def chunk_pdf(file_path, chunk_size=1000):
    pdf = pypdf.PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    
    return chunks

if __name__ == "__main__":
    file_path = Path(__file__).parent.parent.parent / "resources" / "Attention-Is-All-You-Need.pdf"
    chunks = chunk_pdf(file_path)
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}:\n{chunk}\n")