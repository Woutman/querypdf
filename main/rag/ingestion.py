import io
import uuid
import asyncio
import json
from enum import Enum
from copy import deepcopy
from typing import Any

import pymupdf
from pydantic import BaseModel
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from google.genai.types import File, GenerateContentResponse

from settings import get_settings
from instructions import INSTRUCTIONS_TEXT_EXTRACTION
from llm.gemini_interface import upload_file_async, query_gemini_async
from database.vector_store import upsert_elements

ingestion_settings = get_settings().ingestion_settings
gemini_settings = get_settings().gemini_settings


async def ingest_pdf_async(pdf_file: UploadedFile) -> None:
    """Main pipeline for ingestion of an uploaded PDF file."""
    pdf_pages = _split_pdf(pdf_file=pdf_file)
    
    uploaded_files = await _upload_pages_async(pdf_pages=pdf_pages)
    extracted_texts = await _extract_text_async(uploaded_files=uploaded_files)
    # Chunk extracted texts with chunk hierarchie for NarrativeTexts
    # Ingest chunks


async def _upload_pages_async(pdf_pages: list[io.BytesIO]) -> list[File]:
    upload_tasks = []
    for i, page in enumerate(pdf_pages):
        upload_tasks.append(upload_file_async(file=page))
    uploaded_files = await asyncio.gather(*upload_tasks)
    return uploaded_files


class Element(BaseModel):
    type: str
    text: str


class Elements(BaseModel):
    elements: list[Element]
    

async def _extract_text_async(uploaded_files: list[File]) -> list[str]:
    response_tasks = []
    for file in uploaded_files:
        # TODO: Add validation to responses
        response_tasks.append(query_gemini_async(
            prompt=INSTRUCTIONS_TEXT_EXTRACTION, 
            model=gemini_settings.default_model, 
            file=file,
            return_json=True,
            json_schema=Elements
        ))
    responses = await asyncio.gather(*response_tasks)

    responses_processed = _process_responses(responses=responses)

    return responses_processed


class ExtractedElementType(Enum):
    TITLE = "Title"
    SUBTITLE = "Subtitle"
    NARRATIVE_TEXT = "NarrativeText"
    LIST = "List"
    TABLE = "Table"
    INFOGRAPHIC = "Infographic"
    GRAPH = "Graph"
    HEADER = "Header"
    FOOTER = "Footer"
    OTHER_TEXT = "OtherText"


def _process_responses(responses: list[GenerateContentResponse]) -> list[str]:
    for response in responses:
        if not response.text:
            raise ValueError("No text found in response.")
        
    responses_deserialized = [json.loads(response.text) for response in responses] # type: ignore
    respones_flattened = [item for response in responses_deserialized for item in response]

    # Remove all irrelevant types from extracted data.
    responses_processed = []
    types_to_process = ["NarrativeText", "List", "Table", "Infographic", "Graph"]
    previous_item = None
    for item in respones_flattened:
        item_type = item.get("type", "")
        if item_type not in types_to_process:
            previous_item = item
            continue
        # Prepend NarrativeTexts and Lists with corresponding Subtitle if available.
        elif item_type in ["NarrativeText", "List"]:
            if previous_item and previous_item.get("type", "") == "Subtitle":
                item["text"] = previous_item.get("text") + "\n"+ item["text"]
        previous_item = item
        responses_processed.append(item)
    
    return responses_processed


def ingest_pdf(pdf_file: UploadedFile) -> None:
    # TODO: Remove old solutions
    elements = _extract_text_elements(pdf_file=pdf_file)
    elements_chunked = _chunk_elements(elements=elements)
    """
    extracted_text = _extract_text(pdf_file=pdf_file)
    chunks = _chunk_text(extracted_text)
    upsert(documents=chunks)
    """


def _split_pdf(pdf_file: UploadedFile) -> list[io.BytesIO]:
    doc = pymupdf.open(pdf_file.read())
    pdf_pages = []
    for page_num in range(len(doc)):
        new_doc = pymupdf.open()
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

        pdf_bytes_io = io.BytesIO()
        new_doc.save(pdf_bytes_io)
        new_doc.close()

        pdf_bytes_io.seek(0)
        pdf_pages.append(pdf_bytes_io)
    print(f"split pdf in {len(pdf_pages)} pages!")
    return pdf_pages


def _extract_text_elements(pdf_file: UploadedFile) -> list[dict[str, Any]]:
    """Extracts all text elements from an uploaded PDF file and enriches them with metadata."""
    elements = partition_pdf(
        file=io.BytesIO(pdf_file.read()),
        strategy='hi_res'
    )
    text_elements = [e.to_dict() for e in elements if e.category in ['Title', 'NarrativeText', 'ListItem']]
    return text_elements


def _chunk_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Checks if the text of elements is larger than the chunk size. 
    If so, it splits the element in elements with smaller texts, while keeping the metadata.
    """
    replace_mapping = {}
    for i, element in enumerate(elements):
        text = element.get('text', "")
        if len(text) <= ingestion_settings.chunk_size:
            continue

        chunks = _chunk_text(text)
        new_elements = []
        for chunk in chunks:
            # Enrich each chunk with metadata from the original element.
            new_element = deepcopy(element)
            new_element['element_id'] = uuid.uuid4()
            new_element['text'] = chunk
            new_elements.append(new_element)
        replace_mapping[i] = new_elements

    # Replace chunked elements with corresponding new elements.
    for j in reversed(replace_mapping.keys()):
        elements[j:j+1] = replace_mapping[j]

    return elements


def _extract_text(pdf_file: UploadedFile) -> str:
    # TODO: Try Gemini 2.0 Flash for text extraction. Or Unstructured.
    extracted_text = ""
    with pymupdf.open(stream=pdf_file.read(), filetype='pdf') as pdf:
        for i, page in enumerate(pdf): # type: ignore
            page_text = page.get_text("text") # type: ignore
            extracted_text += page_text + "\n"
    return extracted_text.replace("\n", "")


def _chunk_text(text: str) -> list[str]:
    """Splits large texts into smaller texts, according to chunk size."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ingestion_settings.chunk_size,
        chunk_overlap=0,
        separators=ingestion_settings.separators
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks


if __name__ == "__main__":
    import os
    fps = [f"test/files/results/result_{i}" for i in range(10, 15)]
    for fp in fps:
        os.rename(fp, fp + ".txt")
    
    # results = _process_responses()
