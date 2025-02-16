import pymupdf
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database.vector_store import upsert

def ingest_pdf(pdf_file: UploadedFile) -> None:
    extracted_text = _extract_text(pdf_file=pdf_file)
    chunks = _chunk_text(extracted_text)
    upsert(documents=chunks)


def _extract_text(pdf_file: UploadedFile) -> str:
    # TODO: Try Gemini 2.0 Flash for text extraction. Or Unstructured.
    extracted_text = ""
    with pymupdf.open(stream=pdf_file.read(), filetype='pdf') as pdf:
        for i, page in enumerate(pdf): # type: ignore
            if i > 10:  # TODO: Remove this when deploying
                continue
            page_text = page.get_text("text") # type: ignore
            extracted_text += page_text + "\n"
    return extracted_text.replace("\n", "")


def _chunk_text(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks


if __name__ == "__main__":
    import io

    fp = "test/files/jaarverslag2023.pdf"
    pdf_buffer = io.BytesIO()
    with pymupdf.open(fp, filetype='pdf') as doc:
        doc.save(pdf_buffer) 
    pdf_buffer.seek(0)

    text = _extract_text(pdf_buffer) # type: ignore
    print(text)

    chunks = _chunk_text(text)
    print(len(chunks))
