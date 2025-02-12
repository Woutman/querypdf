import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from main.settings import get_settings
from main.llm.openai_interface import query_gpt, get_embeddings
from main.database.vector_store import vec_store
from main.rag.instructions import INSTRUCTIONS_CLASSIFICATION, INSTRUCTIONS_REPHRASING, INSTRUCTIONS_SUMMARIZATION

rag_settings = get_settings().rag_settings


def generate_answer(message_history: list[dict[str, str]]) -> str:
    query = _rephrase_query(message_history=message_history)
    message_history[-1]['content'] = query

    if not _is_rag_necessary(message_history=message_history):
        return _generate_normal_answer(message_history=message_history)

    results = _retrieve_documents(query=query, top_n=rag_settings.top_n_retrieval, min_score=0.0)
    results = _rerank_documents(query=query, documents=results, top_n=rag_settings.top_n_reranking, min_score=0.0)
    results = _reverse_documents(documents=results)
    response = _summarize_documents(query=query, documents=results)
    
    return response


def _is_rag_necessary(message_history: list[dict[str, str]]) -> bool:
    messages = [
        {"role": "system", "content": INSTRUCTIONS_CLASSIFICATION},
        {"role": "user", "content": json.dumps(message_history)}
    ]

    result = query_gpt(messages=messages, temperature=0.0)

    if result == "YES":
        return True
    elif result == "NO":
        return False
    else:
        raise ValueError("Classification step of RAG pipeline returned something other than 'YES' or 'NO'.")


def _generate_normal_answer(message_history: list[dict[str, str]]) -> str:
    response = query_gpt(messages=message_history)
    return response
    

def _rephrase_query(message_history: list[dict[str, str]]) -> str:
    messages = [
        {"role": "system", "content": INSTRUCTIONS_REPHRASING},
        {"role": "user", "content": json.dumps(message_history)}
    ]

    result = query_gpt(messages=messages, temperature=0.0)

    return result


def _retrieve_documents(query: str, top_n: int, min_score: float) -> list[str]:
    query_embeddings = get_embeddings(text=query)
    results = vec_store.search(query_embedding=query_embeddings, limit=top_n)
    if not (documents := results['documents']) or not (distances := results['distances']):
        return []

    documents = documents[0]
    distances = distances[0]

    relevant_documents = _filter_documents_by_distance(documents=documents, distances=distances, min_distance=min_score)

    return relevant_documents


def _rerank_documents(query: str, documents: list[str], top_n: int, min_score: float) -> list[str]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name_or_path = "Alibaba-NLP/gte-multilingual-reranker-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        revision="815b4a86b71f0ecba053e5814a6c24aa7199301e", 
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.to(device)
    model.eval()

    pairs = [[query, doc] for doc in documents]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

    ranked_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, scores in ranked_documents[:top_n]]


def _reverse_documents(documents: list[str]) -> list[str]:
    documents.reverse()
    return documents


def _summarize_documents(query: str, documents: list[str]) -> str:
    documents_as_str = "\n\n".join(documents)
    messages = [
        {"role": "system", "content": INSTRUCTIONS_SUMMARIZATION},
        {"role": "user", "content": f"Documents:\n{documents_as_str}\n\nQuery: {query}"}
    ]

    result = query_gpt(messages=messages, temperature=0.0)

    return result


def _filter_documents_by_distance(documents: list[str], distances: list[float], min_distance: float) -> list[str]:
    indices_above_min_distance = [distances.index(distance) for distance in distances if distance >= min_distance]
    relevant_documents = [doc for doc in documents if documents.index(doc) in indices_above_min_distance]

    return relevant_documents
