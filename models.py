from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict

class Ticket(BaseModel):
    id: int
    text: str

class Observation(BaseModel):
    tickets: List[Ticket]
    current_ticket_id: Optional[int]
    history: List[Dict[str, str]]

class Action(BaseModel):
    action_type: str
    ticket_id: int
    content: Optional[str] = None
    
    classification: Optional[str] = None
    priority: Optional[str] = None
    extracted_fields: Dict[str, str] = Field(default_factory=dict)
    escalate: bool = False
    ask_for_info: bool = False

class Reward(BaseModel):
    score: float
    feedback: str
    
    @field_validator('score')
    @classmethod
    def score_must_be_valid(cls, v):
        return round(max(min(float(v), 0.99), 0.01), 2)