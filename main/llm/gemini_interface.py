import time
import io
import logging
from typing import Optional

from google import genai
from google.genai.types import File, UploadFileConfig, GenerateContentResponse, SchemaUnionDict, GenerateContentConfig
from google.genai.errors import ClientError

from settings import get_settings

gemini_settings = get_settings().gemini_settings
client = genai.Client(api_key="AIzaSyDpsqTSxoHIRTEgYyKG9YJ6EHkwcs1Gh4c")

        
async def upload_file_async(file: io.BytesIO) -> File:
    uploaded_file = await client.aio.files.upload(file=file, config=UploadFileConfig(mime_type='application/pdf'))
    logging.info(f"uploaded file: {uploaded_file.name}")
    return uploaded_file


async def query_gemini_async(
    prompt: str, 
    model: str = gemini_settings.default_model,
    temperature: float = gemini_settings.temperature,
    top_p: float = gemini_settings.top_p, 
    file: Optional[File] = None,
    return_json: bool = False,
    json_schema: Optional[SchemaUnionDict] = None
) -> GenerateContentResponse:
    try:
        response = await client.aio.models.generate_content(
            model=model, 
            contents=[prompt, file] if file else prompt,
            config=GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                response_mime_type='application/json' if return_json else None,
                response_schema=json_schema, 
            ) 
        )
    except ClientError as e:
        # Retry when rate limit is reached.
        if e.code != 429:
            raise e
        sleep_time = 1
        logging.info(f"Rate limit reached. Retrying in: {sleep_time}s.")
        time.sleep(sleep_time)
        response = await query_gemini_async(prompt=prompt, model=model, file=file)

    if not response.text:
        raise ValueError("Gemini returned no output.")
    
    return response
