# Copyright 2024 Philippe Page and TaskFlowAI Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pydantic import BaseModel, Field
from typing import Optional, Callable, Set

class Agent(BaseModel):
    role: str = Field(..., description="The role or type of agent performing tasks")
    goal: str = Field(..., description="The objective or purpose of the agent")
    attributes: Optional[str] = Field(None, description="Additional attributes or characteristics of the agent")
    llm: Optional[Callable] = Field(None, description="The language model function to be used by the agent")
    tools: Optional[Set[Callable]] = Field(default=None, description="Optional set of tool functions")
    temperature: Optional[float] = Field(default=0.7, description="The temperature for the language model")
    max_tokens: Optional[int] = Field(default=4000, description="The maximum number of tokens for the language model")

    model_config = {
        "arbitrary_types_allowed": True
    }