from pydantic import BaseModel, Field
from typing import Optional, Callable

class Agent(BaseModel):
    role: str = Field(..., description="The role or type of agent performing tasks")
    goal: str = Field(..., description="The objective or purpose of the agent")
    attributes: Optional[str] = Field(None, description="Additional attributes or characteristics of the agent")
    llm: Optional[Callable] = Field(None, description="The language model function to be used by the agent")

    class Config:
        arbitrary_types_allowed = True
