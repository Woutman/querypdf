import json
import asyncio
import logging

from google.genai.types import GenerateContentResponse
from google.genai.types import File

from settings import get_settings
from llm.gemini_interface import query_gemini_async
from .instructions import INSTRUCTIONS_TEXT_EXTRACTION
from .types import ExtractedElements

gemini_settings = get_settings().gemini_settings


async def extract_elements_async(uploaded_files: list[File]) -> list[dict[str, str]]:
    """Extracts relevant elements (Texts, Tables, Graphs, etc.) from a list of uploaded PDF files asynchronously."""
    response_tasks = []
    for file in uploaded_files:
        response_tasks.append(_extract_elements_from_file_async(file))
    responses = await asyncio.gather(*response_tasks)

    logging.info(f"Resonses received: {len(responses)}.")
    responses_processed = _process_responses(responses=responses)

    return responses_processed


async def _extract_elements_from_file_async(file: File) -> GenerateContentResponse | None:
    """Extracts relevant elements (Texts, Tables, Graphs, etc.) from an uploaded PDF file asynchronously and retries when a response is invalid."""
    response = await query_gemini_async(
        prompt=INSTRUCTIONS_TEXT_EXTRACTION, 
        model=gemini_settings.default_model, 
        file=file,
        return_json=True,
        json_schema=ExtractedElements
    )

    if not _response_is_valid(response=response):
        # TODO: Add more thorough retry logic that also catches API errors.
        logging.info(f"Retrying extraction for file: {file.name}.")
        return await _extract_elements_from_file_async(file=file)
    else:
        return response


def _response_is_valid(response: GenerateContentResponse) -> bool:
    """Checks whether a Gemini response is valid.""" 
    text = response.text
    if not text:
        return False
    
    text = text.replace("```json", "").replace("```", "")
    try:
        text_deserialized = json.loads(text)
    except:
        return False
    
    if not "elements" in text_deserialized:
        return False
    if not text_deserialized["elements"]:
        return False
    
    for element in text_deserialized["elements"]:
        if "text" not in element or "type" not in element:
            return False
        if not element["type"]:
            return False
        
    return True


def _process_responses(responses: list[GenerateContentResponse]) -> list[dict[str, str]]:
    """Processes a list of Gemini responses into a format that is expected for ingestion."""
    responses_cleaned = [_clean_response(response) for response in responses]
    responses_deserialized = [json.loads(response) for response in responses_cleaned]

    # Remove all irrelevant types from extracted data and flatten list of elements.
    relevant_types = ["NarrativeText", "List", "Table", "Infographic", "Graph", "Subheading"]
    respones_flattened = [item for response in responses_deserialized for item in response.get("elements") if item.get("type", "") in relevant_types]

    # Process elements 
    responses_processed = []
    types_to_process = ["NarrativeText", "List", "Table", "Infographic", "Graph"]
    previous_item = None
    for item in respones_flattened:
        item_type = item.get("type", "")
        if item_type not in types_to_process:
            previous_item = item
            continue
        
        elif item_type in ["NarrativeText", "List"]:
            # Prepend NarrativeTexts and Lists with corresponding Subheading if available.
            if previous_item and previous_item.get("type", "") == "Subheading":
                item["text"] = "##" + previous_item.get("text") + "\n"+ item["text"]
            # Merge NarrativeTexts that are part of sections that span across pages and Lists that have been separated by extraction.
            if previous_item and previous_item.get("type", "") == item_type:
                item["text"] = previous_item["text"] + "\n\n" + item["text"]
                responses_processed.remove(previous_item)
            # Append Lists to preceeding NarrativeTexts, since they are part of the text.
            if previous_item and item_type == "List" and previous_item.get("type", "") == "NarrativeText":
                item["text"] = previous_item["text"] + "\n\n" + item["text"]
                item["type"] = "NarrativeText"
                responses_processed.remove(previous_item)

        previous_item = item
        responses_processed.append(item)
    
    logging.info(f"Responses processed into {len(responses_processed)} elements.")
    return responses_processed


def _clean_response(response: GenerateContentResponse) -> str:
    """Cleans a Gemini response so it can be processed without errors."""
    if not response.text:
        raise ValueError("No text in response.")
    
    response_cleaned = {}
    text_deserialized = json.loads(response.text.replace("```json", "").replace("```", ""))

    response_cleaned["elements"] = [element for element in text_deserialized["elements"] if element["text"]]
    
    response_cleaned = json.dumps(response_cleaned)

    return response_cleaned
