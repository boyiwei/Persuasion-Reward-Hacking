import json
import os
import re
import time
from typing import Union

import attrs
import requests

from openai import OpenAI

# google-genai is imported lazily so this module imports without it. AnthropicChatModel uses the
# OpenAI client against the Anthropic base_url, so the `anthropic` SDK is not needed.

from agents.model.config import ModelAPIProtocol

OpenAIChatPrompt = list[dict[str, str]]
OpenAIBasePrompt = Union[str, list[str]]

PORT_MAP = {
    "Llama3.1-8b-mixed": 30000,
    "Llama3.1-8b-baseline": 29999,
    "Mistral-7b-mixed": 30002,
    "Mistral-7b-baseline": 29998,
    "Qwen2.5-7b-mixed": 30003,
    "Qwen2.5-7b-baseline": 29997,
}

@attrs.define
class OpenAIModel(ModelAPIProtocol):
    # **sampling_kwargs pass from ModelAPI.__call__ through to the API call; with none, no
    # sampling key is sent. Callers that need stochastic sampling pass temperature explicitly.
    def _process_prompt(self, model_id: str, prompt: OpenAIChatPrompt):
        return prompt

    def _make_api_call(self, model_id: str, prompt: OpenAIChatPrompt, **sampling_kwargs):
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        api_response = client.chat.completions.create(
            model=model_id,
            messages=prompt,
            **sampling_kwargs,
        )
        return api_response.choices[0].message.content

    def __call__(self, model_id: str, prompt: OpenAIChatPrompt, max_attempts: int = 5, **sampling_kwargs):
        for attempt in range(max_attempts):
            try:
                response = self._make_api_call(model_id, prompt, **sampling_kwargs)
                return response
            except Exception as e:
                print(f"Attempt: {attempt + 1} time(s) failed")
                print(f"Error: {e}")
                if attempt == max_attempts - 1:
                    raise e
                time.sleep(2 ** (attempt + 2))

class OpenAIBaseModel(OpenAIModel):
    def _process_prompt(self, model_id: str, prompt: OpenAIBasePrompt):
        return [{"role": "user", "content": prompt}]

    def _make_api_call(self, model_id: str, prompt: OpenAIChatPrompt, **sampling_kwargs):
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        api_response = client.chat.completions.create(
            model=model_id,
            messages=prompt,
            **sampling_kwargs,
        )
        return api_response.choices[0].message.content

class OpenAIChatModel(OpenAIModel):
    def _process_prompt(self, model_id: str, prompt: OpenAIChatPrompt):
        return prompt

    def _make_api_call(self, model_id: str, prompt: OpenAIChatPrompt, **sampling_kwargs):
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        api_response = client.chat.completions.create(
            model=model_id,
            messages=prompt,
            **sampling_kwargs,
        )
        return api_response.choices[0].message.content

class NVIDIAChatModel(OpenAIModel):
    def _process_prompt(self, model_id: str, prompt: OpenAIChatPrompt):
        return prompt

    def _make_api_call(self, model_id: str, prompt: OpenAIChatPrompt, **sampling_kwargs):
        nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
        )
        api_response = nvidia_client.chat.completions.create(
            model=model_id,
            messages=prompt,
            **sampling_kwargs,
        )
        return api_response.choices[0].message.content

class OpenRouterChatModel(OpenAIModel):
    def _process_prompt(self, model_id: str, prompt: OpenAIChatPrompt):
        return prompt

    def _make_api_call(self, model_id: str, prompt: OpenAIChatPrompt, **sampling_kwargs):
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        api_response = openrouter_client.chat.completions.create(
            model=model_id,
            messages=prompt,
            **sampling_kwargs,
        )
        return api_response.choices[0].message.content

class GatewayChatModel(OpenAIModel):
    """The API gateway (OpenAI-compatible chat completions plus a `tier` knob).

    The URL is GATEWAY_URL (no default), auth is GATEWAY_API_KEY and tier is GATEWAY_TIER (default
    "base"), all read at call time. An unset GATEWAY_URL raises before any attempt; HTTP errors
    raise, so OpenAIModel.__call__ retries them.
    """
    def __call__(self, model_id: str, prompt: OpenAIChatPrompt, max_attempts: int = 5, **sampling_kwargs):
        if not os.getenv("GATEWAY_URL"):
            raise RuntimeError(
                f"GATEWAY_URL is unset or empty, so gateway model {model_id!r} has no endpoint. "
                "Export it before running, set to the API gateway's OpenAI-compatible "
                "chat-completions URL.")
        return super().__call__(model_id, prompt, max_attempts=max_attempts, **sampling_kwargs)

    def _process_prompt(self, model_id: str, prompt: OpenAIChatPrompt):
        return prompt

    def _make_api_call(self, model_id: str, prompt: OpenAIChatPrompt, **sampling_kwargs):
        # sampling_kwargs go into the POST body
        response = requests.post(
            os.environ["GATEWAY_URL"],
            headers={
                "Authorization": f"Bearer {os.getenv('GATEWAY_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "tier": os.getenv("GATEWAY_TIER", "base"),
                "messages": prompt,
                **sampling_kwargs,
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class AnthropicChatModel(OpenAIModel):
    def _process_prompt(self, model_id: str, prompt: OpenAIChatPrompt):
        return prompt

    def _make_api_call(self, model_id: str, prompt: OpenAIChatPrompt, **sampling_kwargs):
        client = OpenAI(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com/v1/"
        )
        # Anthropic requires max_tokens; pop so an override is not passed twice.
        max_tokens = sampling_kwargs.pop("max_tokens", 1000)
        api_response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=prompt,
            **sampling_kwargs,
        )
        return api_response.choices[0].message.content
    
class GeminiModel(OpenAIModel):

    def __init__(self):
        self.client = None  # Lazy initialization

    def _get_client(self):
        """Lazy initialization of Gemini client"""
        if self.client is None:
            from google import genai
            self.client = genai.Client()
        return self.client

    def _process_prompt(self, model_id: str, prompt: OpenAIChatPrompt):
        # Extract system and user messages
        system_content = None
        user_content = None
        for message in prompt:
            if message["role"] == "system":
                system_content = message["content"]
            elif message["role"] == "user":
                user_content = message["content"]
        return system_content, user_content

    def _make_api_call(self, model_id: str, prompt: OpenAIChatPrompt, **sampling_kwargs):
        from google.genai import types

        system_content, user_content = self._process_prompt(model_id, prompt)

        # map OpenAI-style sampling kwargs onto GenerateContentConfig; unset keys keep defaults
        config_kwargs = {"system_instruction": system_content}
        if "temperature" in sampling_kwargs:
            config_kwargs["temperature"] = sampling_kwargs["temperature"]
        if "max_tokens" in sampling_kwargs:
            config_kwargs["max_output_tokens"] = sampling_kwargs["max_tokens"]

        client = self._get_client()  # Initialize only when needed
        response = client.models.generate_content(
            model=model_id,
            config=types.GenerateContentConfig(**config_kwargs),
            contents=user_content
        )

        return response.text

def _sglang_endpoint(model_id: str):
    """Resolve the SGLang host/port for `model_id`, read at call time.

    SGLANG_HOST_<KEY> / SGLANG_PORT_<KEY> (KEY = model_id upper-cased, non-alphanumerics -> '_')
    override SGLANG_HOST / SGLANG_PORT (default 127.0.0.1:30001), so one process can reach several
    servers (e.g. the RL sender and the Qwen3.5-35B receiver/judge).
    """
    key = re.sub(r"[^A-Z0-9]", "_", model_id.upper())
    host = os.getenv(f"SGLANG_HOST_{key}") or os.getenv("SGLANG_HOST", "127.0.0.1")
    port = os.getenv(f"SGLANG_PORT_{key}") or os.getenv("SGLANG_PORT", "30001")
    return host, port


# SGLang sampling knobs that are NOT OpenAI-standard and must ride extra_body (cf. rl/receiver_client).
_SGLANG_EB_KEYS = ("top_k", "min_p", "repetition_penalty")


def _sglang_sampling(model_id: str) -> dict:
    """Per-model sampling override from JSON env var SGLANG_SAMPLING_<KEY>, or {} if unset/invalid.

    KEY as in _sglang_endpoint.

    Lets an eval sample like the RL rollout (e.g. Qwen3.5 receiver/judge temp=1.0/top_p=0.95/
    top_k=20/presence_penalty=1.5, policy temp=1.0/top_p=1.0) without threading kwargs through
    every agent call."""
    key = re.sub(r"[^A-Z0-9]", "_", model_id.upper())
    raw = os.getenv(f"SGLANG_SAMPLING_{key}")
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return d if isinstance(d, dict) else {}


class SGLangModel(OpenAIModel):
    def _process_prompt(self, model_id: str, prompt: OpenAIChatPrompt):
        return prompt

    def _make_api_call(self, model_id: str, prompt: OpenAIChatPrompt, **sampling_kwargs):
        host, port = _sglang_endpoint(model_id)
        sglang_client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="None")
        # SGLANG_SEED (optional): forwarded to the sampler for best-effort reproducibility;
        # temperature stays at the model default so seeded runs differ from unseeded ones only in
        # the seed.
        extra = {}
        seed = os.getenv("SGLANG_SEED")
        if seed not in (None, ""):
            extra["seed"] = int(seed)
        # SGLANG_SAMPLING_<KEY>: OpenAI fields top-level, SGLang-only knobs in extra_body (as in
        # rl/receiver_client.chat).
        samp = _sglang_sampling(model_id)
        if samp:
            eb = dict(extra.get("extra_body") or {})
            for k, v in samp.items():
                if k in _SGLANG_EB_KEYS:
                    eb[k] = v
                else:
                    extra[k] = v
            if eb:
                extra["extra_body"] = eb
        # caller-supplied sampling kwargs win
        extra.update(sampling_kwargs)
        # SGLANG_ENABLE_THINKING (optional, 0/1): sets chat_template_kwargs.enable_thinking, e.g. 0
        # for concise Qwen3-8B answers; unset uses the server default. Serve with
        # '--reasoning-parser qwen3' so any (even empty) <think> is stripped from content.
        think = os.getenv("SGLANG_ENABLE_THINKING")
        if think not in (None, ""):
            enable = think.strip().lower() not in ("0", "false", "off", "no")
            extra_body = dict(extra.get("extra_body") or {})
            ctk = dict(extra_body.get("chat_template_kwargs") or {})
            ctk["enable_thinking"] = enable
            extra_body["chat_template_kwargs"] = ctk
            extra["extra_body"] = extra_body
        response = sglang_client.chat.completions.create(
            model=model_id,
            messages=prompt,
            **extra,
        )
        return response.choices[0].message.content