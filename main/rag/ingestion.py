import io
import asyncio
import logging

import pymupdf
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.genai.types import File

from settings import get_settings
from llm.gemini_interface import upload_file_async
from database.context_store import insert_context_data
from database.vector_store import upsert_sections_async
from .extraction import extract_elements_async
from .types import Chunk, Paragraph, Section

ingestion_settings = get_settings().ingestion_settings
gemini_settings = get_settings().gemini_settings


async def ingest_pdf_async(pdf_file: UploadedFile) -> None:
    """
    Main pipeline for ingestion of an uploaded PDF file. It performs the following steps:
        1. The PDF file is split into pages.
        2. The pages are uploaded to Gemini.
        3. Relevant elements (Paragraphs, Tables, Graphs, etc.) are extracted from the pages.
        4. Text elements are hierarchically divided into chunks. This hierarchy is the chunk context.
        5. The context is inserted in the context store.
        6. The lowest level chunks are upserted into the vector store as docments.
    """
    import time
    start_time = time.perf_counter()

    pdf_pages = _split_pdf(pdf_file=pdf_file)
    
    uploaded_files = await _upload_pages_async(pdf_pages=pdf_pages)
    extracted_elements = await extract_elements_async(uploaded_files=uploaded_files)
    elements_chunked = _chunk_elements(elements=extracted_elements)
    insert_context_data(elements_chunked)
    await upsert_sections_async(elements_chunked)

    end_time = time.perf_counter()
    logging.info(f"PDF ingested in {end_time-start_time} seconds.")
    

def _split_pdf(pdf_file: UploadedFile) -> list[io.BytesIO]:
    """Splits a PDF file into its individual pages.""" 
    doc = pymupdf.open(stream=pdf_file.read(), filetype='pdf')
    pdf_pages = []
    for page_num in range(len(doc)):
        new_doc = pymupdf.open()
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

        pdf_bytes_io = io.BytesIO()
        new_doc.save(pdf_bytes_io)
        new_doc.close()

        pdf_bytes_io.seek(0)
        pdf_pages.append(pdf_bytes_io)
    logging.info(f"split pdf in {len(pdf_pages)} pages!")
    return pdf_pages


async def _upload_pages_async(pdf_pages: list[io.BytesIO]) -> list[File]:
    """Uploads a PDF file split into individual pages to Gemini asynchronously."""
    upload_tasks = []
    for page in pdf_pages:
        upload_tasks.append(upload_file_async(file=page))
    uploaded_files = await asyncio.gather(*upload_tasks)
    return uploaded_files


def _chunk_elements(elements: list[dict[str, str]]) -> list[Section]:
    """Divides text elements into smaller chunks and creates a hierarchical structure for hierarchical retrieval."""
    elements_chunked = []
    for element in elements:
        element_type = element["type"]
        if element_type != "NarrativeText":
            section = Section(paragraphs=[
                Paragraph(
                    section_index=0, 
                    chunks=[Chunk(text=element["text"], type=element_type, paragraph_index=0)]
                )
            ])
            elements_chunked.append(section)
            continue
        if element["text"].find("\n\n") == -1 and len(element) <= ingestion_settings.chunk_size:
            section = Section(paragraphs=[
                Paragraph(
                    section_index=0, 
                    chunks=[Chunk(text=element["text"], type=element_type, paragraph_index=0)]
                )
            ])
            elements_chunked.append(section)
            continue
        
        full_text = element["text"]
        paragraphs = full_text.split("\n\n")

        # Remove empty strings
        paragraphs = [paragraph for paragraph in paragraphs if paragraph]

        paragraphs_chunked = _split_paragraphs(paragraphs=paragraphs)
        
        element_chunked = Section(paragraphs=paragraphs_chunked)
        elements_chunked.append(element_chunked)
    logging.info("Chunking of elements completed.")
    return elements_chunked


def _split_paragraphs(paragraphs: list[str]) -> list[Paragraph]:
    """Splits paragraphs in chunks if paragraph is bigger than the chunk size."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ingestion_settings.chunk_size,
        chunk_overlap=0,
        separators=ingestion_settings.separators
    )
    paragraphs_chunked = []
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph) <= ingestion_settings.chunk_size:
            paragraph_chunks = [paragraph]
        else:
            paragraph_chunks = text_splitter.split_text(paragraph)

        paragraph_chunked = Paragraph(
            section_index=i,
            chunks=[Chunk(text=chunk, type="NarrativeText", paragraph_index=j) for j, chunk in enumerate(paragraph_chunks)]
        )
        paragraphs_chunked.append(paragraph_chunked)
    return paragraphs_chunked
