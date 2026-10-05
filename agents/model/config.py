from typing import Dict, List, Protocol, Optional

from pydantic import BaseModel


class PromptConfig(BaseModel):
    partials: Dict[str, str]
    word_limit: int
    behavioral_parameters: Optional[Dict[str, float]] = None
    messages: List[Dict[str, str]]
    
class LanguageModelConfig(BaseModel):
    model_id: str
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 1000
    num_candidates: int = 1
    timeout: int = 120
    
class ModelAPIProtocol(Protocol):
    def __call__(self, model_id: str, prompt: str):
        raise NotImplementedError