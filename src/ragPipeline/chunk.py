import pypdf
from pathlib import Path
import re


def _chunk_from_pages(pages, source_name, chunk_size=1000, overlap=200):
    """Core chunking: flatten (page_num, text) pairs then split into overlapping sentence chunks."""
    full_text = ""
    char_to_page = []
    for page_num, text in pages:
        full_text += text
        char_to_page.extend([page_num] * len(text))

    if not full_text.strip():
        return []

    # lookbehind on .!? keeps punctuation attached to the sentence that ends with it
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
            "source": source_name,
            "page": page,
            "chunk_index": len(chunks),
        })

        # find overlap: step back from j until we've covered ~overlap chars
        overlap_len = 0
        step_back = j - 1
        while step_back > i and overlap_len < overlap:
            overlap_len += len(sentences[step_back]) + 1
            step_back -= 1
        i = step_back + 1

    return chunks


# ── Extractors ────────────────────────────────────────────────────────────────

def _extract_pdf(file_path):
    """Extract text page-by-page from a PDF."""
    pdf = pypdf.PdfReader(file_path)
    return [(n + 1, page.extract_text() or "") for n, page in enumerate(pdf.pages)]


def _extract_plain_text(file_path):
    """Read a plain text file as a single page."""
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    return [(1, text)]


def _extract_structured_text(file_path):
    """Read structured text formats (JSON, YAML, TOML, XML) as a single page of raw text."""
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    return [(1, text)]


_EXTRACTORS = {
    ".pdf":  _extract_pdf,
    ".txt":  _extract_plain_text,
    ".md":   _extract_plain_text,
    ".log":  _extract_plain_text,
    ".rst":  _extract_plain_text,
    ".ini":  _extract_plain_text,
    ".cfg":  _extract_plain_text,
    ".conf": _extract_plain_text,
    ".toml": _extract_structured_text,
    ".yaml": _extract_structured_text,
    ".yml":  _extract_structured_text,
    ".json": _extract_structured_text,
    ".xml":  _extract_structured_text,
}

# Exported so other modules can validate without importing private internals
SUPPORTED_EXTENSIONS = frozenset(_EXTRACTORS)


def chunk_file(file_path, chunk_size=1000, overlap=200):
    """Chunk any supported file into overlapping sentence-aware chunks with source and page metadata."""
    ext = Path(file_path).suffix.lower()
    if ext not in _EXTRACTORS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    pages = _EXTRACTORS[ext](file_path)
    return _chunk_from_pages(pages, Path(file_path).name, chunk_size, overlap)


# kept for backward compatibility
def chunk_pdf(file_path, chunk_size=1000, overlap=200):
    """Split a PDF into overlapping sentence-aware chunks with source and page metadata."""
    return chunk_file(file_path, chunk_size, overlap)


if __name__ == "__main__":
    file_path = Path(__file__).parent.parent.parent / "resources" / "Attention-Is-All-You-Need.pdf"
    chunks = chunk_file(file_path)
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}:\n{chunk}\n")
