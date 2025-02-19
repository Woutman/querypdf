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
    file: Optional[File] = None,
    return_json: bool = False,
    json_schema: Optional[SchemaUnionDict] = None
) -> GenerateContentResponse:
    try:
        response = await client.aio.models.generate_content(
            model=model, 
            contents=[prompt, file] if file else prompt,
            config=GenerateContentConfig(
                temperature=0.0,
                top_p=0.95,
                response_mime_type='application/json',
                response_schema=json_schema, 
            ) if return_json else None
        )
    except ClientError as e:
        if e.code != 429:
            raise e
        time.sleep(1)
        response = await query_gemini_async(prompt=prompt, model=model, file=file)

    if not response.text:
        raise ValueError("Gemini returned no output.")
    
    return response
