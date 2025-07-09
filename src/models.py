from langchain_core.documents import Document

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import re

class Option(BaseModel):
  option: str
  is_correct: bool

class Question(BaseModel):
  title: str
  description: str
  options: List[Option]

class Filter(BaseModel):
  certification_id: Optional[str] = None
  tech: Optional[str] = None
  tech_id: Optional[str] = None
  training_slug: Optional[str] = None

class ExplainRequest(BaseModel):
  question: Question
  filter: Filter = Field(default_factory=Filter)

class OptionExplanation(BaseModel):
  option: str
  is_correct: bool
  explanation: str

class TrainingReference(BaseModel):
  """A reference to a specific training document with URL for navigation"""
  title: str
  chapter: str
  training_slug: str
  tech: str
  url: str
  content: str
  similarity_score: float

class ExplanationState(BaseModel):
  context: str = ""
  docs_by_chapter: dict[str, list[Document]] = {}
  training_references: List[TrainingReference] = []

  question: Question
  current_option_index: int = 0
  current_answer: Optional[str] = None

  current_review: Optional[str] = None
  max_retries: int = 3

  option_explanations: List[OptionExplanation] = []
  is_complete: bool = False
  
  certification_id: Optional[str] = None
  tech: Optional[str] = None
  tech_id: Optional[str] = None
  training_slug: Optional[str] = None


class ReviewAnswerResponse(BaseModel):
  is_approved: bool
  review: str

class AnswerQuestionResponse(BaseModel):
  explanation: str = Field(description="The list of bullets of explanation points, without introductions or unnecessary text, just the list string text")

class ReformulateContextResponse(BaseModel):
  reformulated_context: str = Field(description="A focused, coherent paragraph that reformulates the training material to directly address the question being asked")
class ReferenceClassification(BaseModel):
  reference_number: int = Field(description="The reference number (1-5)")
  classification: str = Field(description="Either 'RELEVANT' or 'IRRELEVANT'")
  reasoning: str = Field(description="Brief reasoning for the classification (1-2 sentences)")

class ClassifyReferencesResponse(BaseModel):
  classifications: List[ReferenceClassification] = Field(description="List of reference classifications")

class ExplanationResponse(BaseModel):
  question: Question
  option_explanations: List[OptionExplanation]
  is_complete: bool
  training_references: List[TrainingReference] = []

def to_kebab_case(text: str) -> str:
  text = re.sub(r'[_\s]+', '-', text.lower())
  text = re.sub(r'[^a-z0-9-]', '', text)
  text = re.sub(r'-+', '-', text)
  return text.strip('-')
