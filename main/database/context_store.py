import uuid
import logging
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID

from settings import get_settings
from .models import Base, ChunkORM, ParagraphORM, SectionORM

context_store_settings = get_settings().context_store_settings
rag_settings = get_settings().rag_settings

engine = create_engine(context_store_settings.service_url)


class UUIDBaseModel(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)


class Chunk(UUIDBaseModel):
    paragraph_index: int
    text: str
    type: str
    embeddings: Optional[list[float]] = None


class Paragraph(UUIDBaseModel):
    section_index: int
    chunks: list[Chunk]


class Section(UUIDBaseModel):
    paragraphs: list[Paragraph]


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(engine)


def insert_context_data(context_data: list[Section]) -> None:
    session = SessionLocal()
    try:
        for context in context_data:
            section_orm = SectionORM(id=context.id)
            for paragraph in context.paragraphs:
                paragraph_orm = ParagraphORM(
                    id=paragraph.id, 
                    section=section_orm,
                    section_index=paragraph.section_index
                )
                chunks_orm = [
                    ChunkORM(
                        id=chunk.id, 
                        paragraph_index=chunk.paragraph_index, 
                        text=chunk.text, 
                        paragraph=paragraph_orm
                    ) for chunk in paragraph.chunks
                ]
            session.add(section_orm)
        session.commit()
        logging.info(f"Context data inserted.")
    except Exception as e:
        session.rollback()
        logging.error(f"Error: {e}")
    finally:
        session.close()


def retrieve_parent_chunks(chunk_ids: list[UUID]) -> list[str]:
    session = SessionLocal()

    # Fetch all relevant chunks
    chunks_query = session.execute(select(ChunkORM).where(ChunkORM.id.in_(chunk_ids)))
    chunks = {chunk.id: chunk for chunk in chunks_query.scalars().all()}
    
    # Group chunks by paragraph
    paragraphs_map = defaultdict(list)
    for chunk in chunks.values():
        paragraphs_map[chunk.paragraph_id].append(chunk)
    
    returned_chunks: list[str] = []
    returned_paragraphs: dict[UUID, ParagraphORM] = {}
    for paragraph_id, included_chunks in paragraphs_map.items():
        paragraph = session.get(ParagraphORM, paragraph_id)
        if not paragraph:
            raise Exception("Paragraph not found in context store.")
        
        all_chunks = paragraph.chunks
        
        if len(included_chunks) / len(all_chunks) >= rag_settings.add_paragraph_threshold:
            returned_paragraphs[paragraph_id] = paragraph  # Include full paragraph
        else:
            # Add chunks individually
            returned_chunks.extend([chunk.text for chunk in included_chunks])
    
    # Group returned paragraphs by section
    sections_map = defaultdict(list)
    for paragraph_id, paragraph in returned_paragraphs.items():
        sections_map[paragraph.section_id].append(paragraph)
    
    returned_sections: list[Section] = []
    for section_id, included_paragraphs in sections_map.items():
        section = session.get(SectionORM, section_id)
        if not section:
            raise Exception("Section not found in context store.")
        
        all_paragraphs = section.paragraphs
        
        if len(included_paragraphs) / len(all_paragraphs) >= rag_settings.add_section_threshold:
            returned_sections.append(section)  # Include full section
        else:
            # Merge chunks of included paragraphs in one bigger chunk and include individually
            paragraph_chunks = []
            for paragraph in included_paragraphs:
                chunks = sorted(paragraph.chunks, key=lambda chunk: chunk.paragraph_index)
                paragraph_chunk = " ".join([chunk.text for chunk in chunks])
                paragraph_chunks.append(paragraph_chunk.replace("  ", " ").replace(" .", "."))
            returned_chunks.extend(paragraph_chunks)
    
    if returned_sections:
        # Merge chunks of paragraphs of sections in one bigger chunk
        for section in returned_sections:
            paragraph_chunks = {}   
            for paragraph in section.paragraphs:
                chunks = sorted(paragraph.chunks, key=lambda chunk: chunk.paragraph_index)
                paragraph_chunk = " ".join([chunk.text for chunk in chunks])
                paragraph_chunks[paragraph.section_index] = (paragraph_chunk.replace("  ", " ").replace(" .", ".")) 
            section_chunk = "\n\n".join(dict(sorted(paragraph_chunks.items())).values())
            returned_chunks.append(section_chunk)

    # Remove chunks that are substrings of other chunks
    result = []
    for chunk in returned_chunks:
        if not any(chunk != returned_chunk and chunk in returned_chunk for returned_chunk in returned_chunks):
            result.append(chunk)

    return result
