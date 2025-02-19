import uuid
import logging

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP, func, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID

from settings import get_settings

context_store_settings = get_settings().context_store_settings

engine = create_engine(context_store_settings.service_url)

Base = declarative_base()


class UUIDBaseModel(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)


class ChunkORM(Base):
    __tablename__ = "chunks"
    id = Column(UUID, primary_key=True, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    text = Column(String, unique=False, nullable=False)

    paragraph_id = Column(UUID, ForeignKey("paragraphs.id", ondelete="CASCADE"), nullable=False)
    paragraph = relationship("ParagraphORM", back_populates="chunks")
    paragraph_index = Column(Integer, nullable=False)


class Chunk(UUIDBaseModel):
    paragraph_index: int
    text: str
    type: str


class ParagraphORM(Base):
    __tablename__ = "paragraphs"
    id = Column(UUID, primary_key=True, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())   

    chunks = relationship("ChunkORM", back_populates="paragraph", cascade="all, delete-orphan")
    section_id = Column(UUID, ForeignKey("sections.id", ondelete="CASCADE"), nullable=False)
    section = relationship("SectionORM", back_populates="paragraphs")
    section_index = Column(Integer, nullable=False)


class Paragraph(UUIDBaseModel):
    section_index: int
    chunks: list[Chunk]


class SectionORM(Base):
    __tablename__ = "sections"
    id = Column(UUID, primary_key=True, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    paragraphs = relationship("ParagraphORM", back_populates="section", cascade="all, delete-orphan")


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
    except Exception as e:
        session.rollback()
        logging.error(f"Error: {e}")
    finally:
        session.close()
