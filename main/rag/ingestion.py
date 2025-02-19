import io
# mport uuid
import asyncio
import json
from enum import Enum
# from copy import deepcopy
# from typing import Any

import pymupdf
from pydantic import BaseModel
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from unstructured.partition.pdf import partition_pdf
from google.genai.types import File, GenerateContentResponse

from settings import get_settings
from .instructions import INSTRUCTIONS_TEXT_EXTRACTION
from llm.gemini_interface import upload_file_async, query_gemini_async
from database.context_store import insert_context_data, Section, Paragraph, Chunk
from database.vector_store import upsert_elements

ingestion_settings = get_settings().ingestion_settings
gemini_settings = get_settings().gemini_settings


async def ingest_pdf_async(pdf_file: UploadedFile) -> None:
    """Main pipeline for ingestion of an uploaded PDF file."""
    pdf_pages = _split_pdf(pdf_file=pdf_file)
    
    uploaded_files = await _upload_pages_async(pdf_pages=pdf_pages)
    extracted_elements = await _extract_elements_async(uploaded_files=uploaded_files)
    elements_chunked = _chunk_elements(elements=extracted_elements)
    insert_context_data(elements_chunked)

    # TODO: Test _process_responses
    # TODO: Chunk extracted texts with chunk hierarchie for NarrativeTexts
    # TODO: Ingest chunks


def _split_pdf(pdf_file: UploadedFile) -> list[io.BytesIO]:
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
    print(f"split pdf in {len(pdf_pages)} pages!")
    return pdf_pages


async def _upload_pages_async(pdf_pages: list[io.BytesIO]) -> list[File]:
    upload_tasks = []
    for i, page in enumerate(pdf_pages):
        upload_tasks.append(upload_file_async(file=page))
    uploaded_files = await asyncio.gather(*upload_tasks)
    return uploaded_files


class ExtractedElement(BaseModel):
    type: str
    text: str


class ExtractedElements(BaseModel):
    elements: list[ExtractedElement]
    

async def _extract_elements_async(uploaded_files: list[File]) -> list[dict[str, str]]:
    response_tasks = []
    for file in uploaded_files:
        # TODO: Add validation to responses
        response_tasks.append(query_gemini_async(
            prompt=INSTRUCTIONS_TEXT_EXTRACTION, 
            model=gemini_settings.default_model, 
            file=file,
            return_json=True,
            json_schema=ExtractedElements
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


def _process_responses(responses: list[GenerateContentResponse]) -> list[dict[str, str]]:

    responses_deserialized = [json.loads(response.text) for response in responses if response.text]
    relevant_types = ["NarrativeText", "List", "Table", "Infographic", "Graph", "Subtitle"]
    respones_flattened = [item for response in responses_deserialized for item in response.get("elements") if item.get("type", "") in relevant_types]

    # Remove all irrelevant types from extracted data.
    responses_processed = []
    types_to_process = ["NarrativeText", "List", "Table", "Infographic", "Graph"]
    previous_item = None
    for item in respones_flattened:
        item_type = item.get("type", "")
        if item_type not in types_to_process:
            previous_item = item
            continue

        # TODO: Add NarrativeTexts to sections that span across pages
        
        elif item_type in ["NarrativeText", "List"]:
            # Prepend NarrativeTexts and Lists with corresponding Subtitle if available.
            if previous_item and previous_item.get("type", "") == "Subtitle":
                item["text"] = "##"+ previous_item.get("text") + "\n"+ item["text"]
            # Merge NarrativeTexts that are part of sections that span across pages.
            if previous_item and previous_item.get("type", "") == "NarrativeText" and item_type == "NarrativeText":
                item["text"] = previous_item["text"] + item["text"]
                responses_processed.remove(previous_item)
        previous_item = item
        responses_processed.append(item)
    
    return responses_processed


def _chunk_elements(elements: list[dict[str, str]]) -> list[Section]:
    elements_chunked = []
    for element in elements:
        element_type = element.get("type", "")
        if element_type != "NarrativeText":
            section = Section(paragraphs=[Paragraph(chunks=[Chunk(text=element.get("text", ""), type=element_type)])])
            elements_chunked.append(section)
            continue
        if element["text"].find("\n\n") == -1 and len(element) <= ingestion_settings.chunk_size:
            section = Section(paragraphs=[Paragraph(chunks=[Chunk(text=element.get("text", ""), type=element_type)])])
            elements_chunked.append(section)
            continue
        
        full_text = element["text"]

        # Split texts in paragraphs.
        paragraphs = full_text.split("\n\n")

        # Split paragraphs in chunks if paragraph is bigger than max chunk size.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ingestion_settings.chunk_size,
            chunk_overlap=0,
            separators=ingestion_settings.separators
        )
        paragraphs_chunked = []
        for paragraph in paragraphs:
            if len(paragraph) <= ingestion_settings.chunk_size:
                paragraph_chunks = [paragraph]
            else:
                paragraph_chunks = text_splitter.split_text(paragraph)        
            paragraph_chunked = Paragraph(chunks=[Chunk(text=chunk, type=element.get("type", "")) for chunk in paragraph_chunks])
            paragraphs_chunked.append(paragraph_chunked)
        
        element_chunked = Section(paragraphs=paragraphs_chunked)
        elements_chunked.append(element_chunked)
    return elements_chunked

'''
def ingest_pdf(pdf_file: UploadedFile) -> None:
    # TODO: Remove old solutions
    elements = _extract_text_elements(pdf_file=pdf_file)
    elements_chunked = _chunk_extracted_elements(elements=elements)
    """
    extracted_text = _extract_text(pdf_file=pdf_file)
    chunks = _chunk_text(extracted_text)
    upsert(documents=chunks)
    """


def _extract_text_elements(pdf_file: UploadedFile) -> list[dict[str, Any]]:
    """Extracts all text elements from an uploaded PDF file and enriches them with metadata."""
    elements = partition_pdf(
        file=io.BytesIO(pdf_file.read()),
        strategy='hi_res'
    )
    text_elements = [e.to_dict() for e in elements if e.category in ['Title', 'NarrativeText', 'ListItem']]
    return text_elements


def _chunk_extracted_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
'''