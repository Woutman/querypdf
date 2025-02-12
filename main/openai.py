from typing import Optional, Any

import openai

from .settings import get_settings

settings = get_settings().openai_settings
client = openai.OpenAI(api_key=settings.api_key)


def query_gpt(
        messages: list[dict[str, Any]], 
        model: str = settings.default_model, 
        return_json: bool = False, 
        json_schema: Optional[dict[str, Any]] = None, 
        temperature: float = settings.temperature, 
        top_p: float = settings.top_p
    ) -> str:
    """Sends a query to GPT and returns the output as a string.""" 

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
        raise ValueError("GPT call returned no output.")

    return output


def get_embeddings(text: str) -> list[float]:
    """Returns the vector embeddings of the input string."""
    return client.embeddings.create(input=[text], model=settings.embeddings_model).data[0].embedding
