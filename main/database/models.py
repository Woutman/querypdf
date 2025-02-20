from sqlalchemy import Column, Integer, String, TIMESTAMP, func, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()


class ChunkORM(Base):
    __tablename__ = "chunks"
    id = Column(UUID, primary_key=True, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    text = Column(String, unique=False, nullable=False)

    paragraph_id = Column(UUID, ForeignKey("paragraphs.id", ondelete="CASCADE"), nullable=False)
    paragraph = relationship("ParagraphORM", back_populates="chunks")
    paragraph_index = Column(Integer, nullable=False)


class ParagraphORM(Base):
    __tablename__ = "paragraphs"
    id = Column(UUID, primary_key=True, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())   

    chunks = relationship("ChunkORM", back_populates="paragraph", cascade="all, delete-orphan")
    section_id = Column(UUID, ForeignKey("sections.id", ondelete="CASCADE"), nullable=False)
    section = relationship("SectionORM", back_populates="paragraphs")
    section_index = Column(Integer, nullable=False)


class SectionORM(Base):
    __tablename__ = "sections"
    id = Column(UUID, primary_key=True, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    paragraphs = relationship("ParagraphORM", back_populates="section", cascade="all, delete-orphan")
