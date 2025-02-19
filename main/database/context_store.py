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


class ChunkORM(Base):
    __tablename__ = "chunks"
    id = Column(UUID, primary_key=True, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    text = Column(String, unique=True, nullable=False)

    paragraph_id = Column(Integer, ForeignKey("paragraphs.id", ondelete="CASCADE"), nullable=False)
    paragraph = relationship("ParagraphORM", back_populates="chunks")


class Chunk(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    text: str
    type: str


class ParagraphORM(Base):
    __tablename__ = "paragraphs"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    chunks = relationship("ChunkORM", back_populates="paragraph", cascade="all, delete-orphan")
    section_id = Column(Integer, ForeignKey("sections.id", ondelete="CASCADE"), nullable=False)
    section = relationship("SectionORM", back_populates="paragraphs")


class Paragraph(BaseModel):
    chunks: list[Chunk]


class SectionORM(Base):
    __tablename__ = "sections"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    paragraphs = relationship("ParagraphORM", back_populates="section", cascade="all, delete-orphan")


class Section(BaseModel):
    paragraphs: list[Paragraph]


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(engine)


def insert_context_data(context_data: list[Section]) -> None:
    session = SessionLocal()
    try:
        for context in context_data:
            section_orm = SectionORM()
            for paragraph in context.paragraphs:
                paragraph_orm = ParagraphORM(section=section_orm)
                chunks_orm = [ChunkORM(id=chunk.id, text=chunk.text, paragraph=paragraph_orm) for chunk in paragraph.chunks]
            session.add(section_orm)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error: {e}")
    finally:
        session.close()
