from typing import Optional, Any

import openai

from settings import get_settings

openai_settings = get_settings().openai_settings
client = openai.OpenAI(api_key=openai_settings.api_key)


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
    # TODO: Make async
    return client.embeddings.create(input=[text], model=openai_settings.embeddings_model).data[0].embedding
