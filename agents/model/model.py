from typing import Union
import os
import yaml

import attrs

from agents.model.openai_llm import OpenAIBaseModel, OpenAIChatModel, AnthropicChatModel, NVIDIAChatModel, OpenRouterChatModel, GatewayChatModel, GeminiModel, SGLangModel

def _load_model_lists():
    """Load the provider -> model-id sets from config/available_models.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'available_models.yaml')
    with open(config_path, 'r') as f:
        models = yaml.safe_load(f)

    return {
        'GOOGLE_MODELS': set(models.get('google', [])),
        'OPENAI_BASE_MODELS': set(models.get('openai_base', [])),
        'OPENAI_CHAT_MODELS': set(models.get('openai_chat', [])),
        'ANTHROPIC_MODELS': set(models.get('anthropic', [])),
        'NVIDIA_MODELS': set(models.get('nvidia', [])),
        'SGLANG_MODELS': set(models.get('sglang', [])),
        'OPENROUTER_MODELS': set(models.get('openrouter', [])),
        'GATEWAY_MODELS': set(models.get('gateway', [])),
    }

_MODEL_LISTS = _load_model_lists()
GOOGLE_MODELS = _MODEL_LISTS['GOOGLE_MODELS']
OPENAI_BASE_MODELS = _MODEL_LISTS['OPENAI_BASE_MODELS']
OPENAI_CHAT_MODELS = _MODEL_LISTS['OPENAI_CHAT_MODELS']
ANTHROPIC_MODELS = _MODEL_LISTS['ANTHROPIC_MODELS']
NVIDIA_MODELS = _MODEL_LISTS['NVIDIA_MODELS']
SGLANG_MODELS = _MODEL_LISTS['SGLANG_MODELS']
OPENROUTER_MODELS = _MODEL_LISTS['OPENROUTER_MODELS']
GATEWAY_MODELS = _MODEL_LISTS['GATEWAY_MODELS']


@attrs.define()
class ModelAPI:
    _openai_base: OpenAIBaseModel = attrs.field(init=False)
    _openai_chat: OpenAIChatModel = attrs.field(init=False)
    _anthropic_chat: AnthropicChatModel = attrs.field(init=False)
    _nvidia_chat: NVIDIAChatModel = attrs.field(init=False)
    _openrouter_chat: OpenRouterChatModel = attrs.field(init=False)
    _gateway_chat: GatewayChatModel = attrs.field(init=False)
    _gemini: GeminiModel = attrs.field(init=False)
    _sglang: SGLangModel = attrs.field(init=False)
    
    def __attrs_post_init__(self):
        self._openai_base = OpenAIBaseModel()
        self._openai_chat = OpenAIChatModel()
        self._anthropic_chat = AnthropicChatModel()
        self._nvidia_chat = NVIDIAChatModel()
        self._openrouter_chat = OpenRouterChatModel()
        self._gateway_chat = GatewayChatModel()
        self._gemini = GeminiModel()
        self._sglang = SGLangModel()
        
    def __call__(self, model_id: Union[str, list[str]], prompt: str | list[dict[str, str]], **sampling_kwargs) -> Union[str, list[str]]:
        # Sampling kwargs (temperature, max_tokens, ...) are forwarded to the provider call.
        def model_id_to_class(mid: str):
            if mid in SGLANG_MODELS:
                return self._sglang
            elif mid in NVIDIA_MODELS:
                return self._nvidia_chat
            elif mid in OPENROUTER_MODELS:
                return self._openrouter_chat
            elif mid in GATEWAY_MODELS:
                return self._gateway_chat
            elif mid in OPENAI_BASE_MODELS:
                return self._openai_base
            elif mid in OPENAI_CHAT_MODELS:
                return self._openai_chat
            elif mid in ANTHROPIC_MODELS:
                return self._anthropic_chat
            elif mid in GOOGLE_MODELS:
                return self._gemini
            else:
                raise ValueError(f"Unsupported: {mid}")
        
        if isinstance(model_id, list):
            return [model_id_to_class(mid)(model_id=mid, prompt=prompt, **sampling_kwargs) for mid in model_id]
        else:
            return model_id_to_class(model_id)(model_id=model_id, prompt=prompt, **sampling_kwargs)
