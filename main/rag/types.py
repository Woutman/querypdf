import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

class ExtractedElementType(Enum):
    TITLE = "Title"
    SUBHEADING = "Subheading"
    NARRATIVE_TEXT = "NarrativeText"
    LIST = "List"
    TABLE = "Table"
    INFOGRAPHIC = "Infographic"
    GRAPH = "Graph"
    HEADER = "Header"
    FOOTER = "Footer"
    RUNNINGHEAD = "RunningHead"
    OTHER_TEXT = "OtherText"


class ExtractedElement(BaseModel):
    type: ExtractedElementType
    text: str


class ExtractedElements(BaseModel):
    elements: list[ExtractedElement]


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
