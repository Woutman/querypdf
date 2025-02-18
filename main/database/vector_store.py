import logging
from datetime import datetime
from typing import Any

import psycopg2
from psycopg2.errors import DuplicateTable
from timescale_vector import client
from timescale_vector.client import uuid_from_time

from settings import get_settings
from llm.openai_interface import get_embeddings

vec_settings = get_settings().vector_store_settings

vec_store = client.Sync(
    service_url=vec_settings.service_url, 
    table_name=vec_settings.table_name, 
    num_dimensions=vec_settings.embedding_dimenstions
)

vec_store.create_tables()
try:
    vec_store.create_embedding_index(client.DiskAnnIndex())
except DuplicateTable:
    pass

# Create keyword search index
index_name = f"idx_{vec_settings.table_name}_contents_gin"
create_index_sql = f"""
CREATE INDEX IF NOT EXISTS {index_name}
ON {vec_settings.table_name} USING gin(to_tsvector('english', contents));
"""
try:
    with psycopg2.connect(vec_settings.service_url) as conn:
        with conn.cursor() as cur:
            cur.execute(create_index_sql)
            conn.commit()
            logging.info(f"GIN index '{index_name}' created or already exists.")
except Exception as e:
    logging.error(f"Error while creating GIN index: {str(e)}")


def upsert(documents: list[str]) -> None:
    """Upserts a list of documents and their embeddings into the vector database."""
    data = []
    for document in documents:
        now = datetime.now()
        uuid = str(uuid_from_time(now))
        metadata = {"created_at": now.isoformat()}
        embeddings = get_embeddings(document)
        data.append((uuid, metadata, document, embeddings))
    
    vec_store.upsert(data)


def upsert_elements(elements: list[dict[str, Any]]) -> None:
    """Preprocesses and upserts elements generated by Unstructered's partition_pdf"""
    data = []
    for element in elements:
        element['metadata']['type'] = element.get('type', '')
        uuid = element.get('element_id')
        metadata = element.get('metadata', {})
        document = element.get('text', '')
        embeddings = get_embeddings(document)
        data.append((uuid, metadata, document, embeddings))
    
    vec_store.upsert(data)


if __name__ == "__main__":
    documents = ["Dogs are brown", "Italy is a country"]
    upsert(documents)

    query = "What color are dogs?"
    query_embeddings = get_embeddings(query)
    print(vec_store.search(query_embeddings))
    
    vec_store.delete_all()
