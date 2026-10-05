from typing import Optional

from pydantic import BaseModel

from agents.model.config import LanguageModelConfig, PromptConfig
from agents.model.model import ModelAPI
from agents.rollout import TranscriptConfig


class AgentConfig(BaseModel):
    language_model: LanguageModelConfig
    prompts: PromptConfig
    agent_type: Optional[str] = "standard"

class AgentBase:
    def __init__(self, config: AgentConfig, api_handler: ModelAPI):
        self.config = config
        self.api_handler = api_handler
        self.partials = self.config.prompts.partials
        self.messages = self.config.prompts.messages
        self.behavioral_parameters = self.config.prompts.behavioral_parameters
    
    def construct_messages(self, _: TranscriptConfig):
        raise NotImplementedError