import logging
import asyncio
from typing import Optional, Any

import openai

from settings import get_settings

openai_settings = get_settings().openai_settings
client = openai.OpenAI(api_key=openai_settings.api_key)
client_async = openai.AsyncOpenAI(api_key=openai_settings.api_key)


def query_gpt(
        messages: list[dict[str, Any]], 
        model: str = openai_settings.default_model, 
        return_json: bool = False, 
        json_schema: Optional[dict[str, Any]] = None, 
        temperature: float = openai_settings.temperature, 
        top_p: float = openai_settings.top_p
    ) -> str:
    """Sends a query to GPT and returns the response as a string.""" 

    if return_json and not json_schema:
        raise ValueError("GPT should return JSON, but no JSON schema was provided.")
    
    response = client.chat.completions.create(
        messages=messages,  # type: ignore
        model=model,
        response_format={"type": "json_schema", "json_schema": json_schema} if return_json else {"type": "text"}, # type: ignore
        temperature=temperature,
        top_p=top_p
    )
    
    print(response.usage)
    output = response.choices[0].message.content
    if not output:
        raise ValueError("GPT call returned no response.")

    return output


def get_embeddings(text: str) -> list[float]:
    """Returns the vector embeddings of the input string."""
    if not text:
        raise ValueError("String to embed is empty.")
    return client.embeddings.create(input=[text], model=openai_settings.embeddings_model).data[0].embedding


async def get_embeddings_async(text: str) -> list[float]:
    """Returns the vector embeddings of the input string asynchronously."""
    if not text:
        raise ValueError("String to embed is empty.")
    
    semaphore = asyncio.Semaphore(50)
    async with semaphore:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    client_async.embeddings.create(input=[text], model=openai_settings.embeddings_model),
                    timeout=10
                )
                logging.info("Created embedding!")
                return response.data[0].embedding
            except (asyncio.TimeoutError, Exception) as e:
                wait_time = 2 ** attempt
                logging.warning(f"Attempt {attempt+1} failed with error: {e}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        logging.error("Max retries exceeded for embedding request.")
        raise Exception("Max retries exceeded for embedding request.")
