from pydantic import BaseModel
from typing import Optional, List

class Ticket(BaseModel):
    id: int
    text: str

class Observation(BaseModel):
    tickets: List[Ticket]
    current_ticket_id: Optional[int]

class Action(BaseModel):
    action_type: str
    ticket_id: int
    content: Optional[str] = None

class Reward(BaseModel):
    score: float
    feedback: str