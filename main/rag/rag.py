import json
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from settings import get_settings
from llm.openai_interface import query_gpt, get_embeddings
from database.vector_store import vec_store
from rag.instructions import INSTRUCTIONS_REPHRASING, INSTRUCTIONS_SUMMARIZATION

rag_settings = get_settings().rag_settings


def generate_answer(message_history: list[dict[str, str]]) -> str:
    query = _rephrase_query(message_history=message_history)
    results = _retrieve_documents(query=query, top_n=rag_settings.top_n_retrieval, max_distance=1.0)
    results = _rerank_documents(query=query, documents=results, top_n=rag_settings.top_n_reranking, min_score=0.0)
    response = _summarize_documents(query=query, documents=results)
    
    return response
    

def _rephrase_query(message_history: list[dict[str, str]]) -> str:
    messages = [
        {"role": "system", "content": INSTRUCTIONS_REPHRASING},
        {"role": "user", "content": json.dumps(message_history)}
    ]

    result = query_gpt(messages=messages, temperature=0.0)

    return result


def _retrieve_documents(query: str, top_n: int, max_distance: float) -> list[list[Any]]:
    query_embeddings = get_embeddings(text=query)
    results = vec_store.search(query_embedding=query_embeddings, limit=top_n) # TODO: Implement filters
    if not results:
        return []

    relevant_documents = [doc for doc in results if doc[-1] <= max_distance]

    return relevant_documents


def _rerank_documents(query: str, documents: list[list[Any]], top_n: int, min_score: float) -> list[list[Any]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name_or_path = "Alibaba-NLP/gte-multilingual-reranker-base"

    # TODO: Fetch model at docker-compose up to prevent slow first response
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        revision="815b4a86b71f0ecba053e5814a6c24aa7199301e",    # Model uses custom files, so use this version to ensure no malicious code.
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.to(device)
    model.eval()

    pairs = [[query, doc[2]] for doc in documents]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

    ranked_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True) # Reverse order improves summarization step slightly. For more info, see: https://arxiv.org/pdf/2407.01219
    
    return [doc for doc, scores in ranked_documents[:top_n]]


def _summarize_documents(query: str, documents: list[list[Any]]) -> str:
    documents_content = [doc[2] for doc in documents]
    documents_as_str = "\n\n".join(documents_content)
    messages = [
        {"role": "system", "content": INSTRUCTIONS_SUMMARIZATION},
        {"role": "user", "content": f"Documents:\n{documents_as_str}\n\nQuery: {query}"}
    ]

    result = query_gpt(messages=messages, temperature=0.0)

    return result
