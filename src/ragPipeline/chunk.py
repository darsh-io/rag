import pypdf
from pathlib import Path
import re

def chunk_pdf(file_path, chunk_size=1000, overlap=200):
    """Split a PDF into overlapping sentence-aware chunks with source and page metadata."""
    pdf = pypdf.PdfReader(file_path)

    pages = []
    for page_num, page in enumerate(pdf.pages):
        pages.append((page_num + 1, page.extract_text() or ""))

    # flatten all pages into one string while recording which page each character belongs to
    full_text = ""
    char_to_page = []
    for page_num, text in pages:
        full_text += text
        char_to_page.extend([page_num] * len(text))

    # lookbehind on .!? keeps the punctuation attached to the sentence that ends with it
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    # record the start character of each sentence so we can map it back to a page number
    sentence_positions = []
    pos = 0
    for sentence in sentences:
        sentence_positions.append(pos)
        pos += len(sentence) + 1  # +1 for the space that was split on

    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = []
        chunk_len = 0

        j = i
        while j < len(sentences) and chunk_len + len(sentences[j]) <= chunk_size:
            chunk_sentences.append(sentences[j])
            chunk_len += len(sentences[j]) + 1
            j += 1

        # if a single sentence exceeds chunk_size, include it anyway
        if not chunk_sentences:
            chunk_sentences.append(sentences[i])
            j = i + 1

        chunk_text = " ".join(chunk_sentences)
        char_pos = sentence_positions[i]
        # clamp to last index in case rounding puts us one past the end
        page = char_to_page[min(char_pos, len(char_to_page) - 1)]

        chunks.append({
            "text": chunk_text,
            "source": Path(file_path).name,
            "page": page,
            "chunk_index": len(chunks)
        })

        # find overlap: step back from j until we've covered ~overlap chars
        overlap_len = 0
        step_back = j - 1
        while step_back > i and overlap_len < overlap:
            overlap_len += len(sentences[step_back]) + 1
            step_back -= 1
        i = step_back + 1

    return chunks

if __name__ == "__main__":
    file_path = Path(__file__).parent.parent.parent / "resources" / "Attention-Is-All-You-Need.pdf"
    chunks = chunk_pdf(file_path)
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}:\n{chunk}\n")
