import re

from agents.agent_base import AgentBase
from agents.rollout import TranscriptConfig

# Verdict parsing for the receiver's `<action>` block across per-model failure modes (harmony
# channel tokens, prose verdicts, decorated tags). Labels: Old Bailey acquit/convict/neutral,
# house-showing attend/decline, nutrition adopt/decline. The scan takes the last keyword in the
# block, and Old Bailey blocks never contain the other domains' words.
_ACTION_LABELS = ("acquit", "convict", "neutral", "attend", "decline", "adopt")
_ACTION_KW_RE = re.compile(r"\b(acquit|convict|neutral|attend|decline|adopt)\b", re.IGNORECASE)
_ACTION_TAG_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_FINAL_CHANNEL_RE = re.compile(r"<\|channel\|>final<\|message\|>")
_TRAILING_CTRL_RE = re.compile(r"<\|(?:end|return|start|channel)\|>")


def _isolate_final_channel(text: str) -> str:
    if "<|channel|>" not in text:
        return text
    matches = list(_FINAL_CHANNEL_RE.finditer(text))
    if matches:
        text = text[matches[-1].end():]
    return _TRAILING_CTRL_RE.split(text)[0]


def parse_action(message: str) -> str:
    """One of _ACTION_LABELS, or '' if no verdict is parseable. Regex only (no LLM)."""
    if not message:
        return ""
    text = _isolate_final_channel(message)
    text = text.replace("[action]", "<action>").replace("[/action]", "</action>")
    for tag in reversed(_ACTION_TAG_RE.findall(text)):
        found = _ACTION_KW_RE.findall(tag)
        if found:
            return found[-1].lower()
    found = _ACTION_KW_RE.findall(text)
    return found[-1].lower() if found else ""


# Closed <think>/<thinking> pairs, or an unterminated open tag to end-of-text (truncated CoT).
_THINKING_RE = re.compile(r"<think(?:ing)?>.*?(?:</think(?:ing)?>|\Z)", re.IGNORECASE | re.DOTALL)


def _last_closed_block(text: str, tag: str):
    """Content of the last `<tag>...</tag>` pair (case-insensitive), or None.

    The last pair, because models such as Qwen3.5 echo the tag names while reasoning.
    """
    low = text.lower()
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    oi = low.rfind(open_t)
    if oi < 0:
        return None
    ci = low.find(close_t, oi + len(open_t))
    if ci < 0:
        return None
    return text[oi + len(open_t):ci].strip()


def response_without_thinking(response: str) -> str:
    """The receiver's answer without chain-of-thought, for later rounds (transcript_history="full").

    Rebuilt from the last `<belief>` + `<argument>` blocks and the parsed `<action>`, since
    reasoning models (e.g. Qwen3.5) emit tens of KB of untagged reasoning that tag-stripping misses.
    Falls back to the thinking-stripped text, then the action label.
    """
    if not response:
        return response
    text = _THINKING_RE.sub("", response)
    parts = []
    belief = _last_closed_block(text, "belief")
    if belief is not None:
        parts.append(f"<belief>\n{belief}\n</belief>")
    argument = _last_closed_block(text, "argument")
    if argument is not None:
        parts.append(f"<argument>\n{argument}\n</argument>")
    action = parse_action(response)
    if action:
        parts.append(f"<action>{action}</action>")
    cleaned = "\n".join(parts).strip()
    if cleaned:
        return cleaned
    stripped = text.strip()
    # never the raw response, which would bring back stripped CoT
    return stripped if stripped else action


def receiver_response_block(response: str) -> str:
    """The receiver's reply as its <belief> + <argument> blocks (no CoT, no <action>), or ''.

    Used by the `receiver_response` transcript mode, mirroring
    rl/persuasion_interaction._receiver_context."""
    if not response:
        return ""
    text = _THINKING_RE.sub("", response)
    parts = []
    belief = _last_closed_block(text, "belief")
    if belief is not None:
        parts.append(f"<belief>\n{belief}\n</belief>")
    argument = _last_closed_block(text, "argument")
    if argument is not None:
        parts.append(f"<argument>\n{argument}\n</argument>")
    return "\n".join(parts).strip()


class AgentQuality(AgentBase):
    
    def get_transcript_string(self, transcript:TranscriptConfig):
        # `transcript_history` sets what each prior round's receiver entry shows (the rollout is
        # stateless, so this string is the model's memory):
        #   "action" (default)  parsed verdict label only
        #   "full"              receiver's prior response without CoT (response_without_thinking)
        #   "receiver_response" sender view: belief + argument in <receiver_response>, framed like
        #                       the RL rollout (rl/persuasion_interaction._RECEIVER_TURN)
        # Set per agent, e.g. ++receiver.transcript_history=full.
        history = getattr(self.config, "transcript_history", "action")
        responses = transcript.responses or []
        if history == "receiver_response":
            total = int(getattr(self, "total_steps", 3) or 3)
            transcript_string = ""
            for i, round in enumerate(transcript.rounds):
                if round.sender is not None:
                    transcript_string += f"Round {i+1} of {total}.\nSender: {round.sender}\n"
                rr = ""
                if i < len(responses) and responses[i].receiver is not None:
                    rr = receiver_response_block(responses[i].receiver)
                if rr:
                    transcript_string += ("The Receiver responded to your argument:\n"
                                          f"<receiver_response>\n{rr}\n</receiver_response>\n")
            return transcript_string.strip()
        transcript_string = ""
        for i, round in enumerate(transcript.rounds):
            transcript_string += f"Round {i+1}:\n"
            if round.sender is not None:
                transcript_string += f"Sender: {round.sender}\n"
            receiver_entry = round.receiver
            if history == "full" and i < len(responses) and responses[i].receiver is not None:
                receiver_entry = response_without_thinking(responses[i].receiver)
            if receiver_entry is not None:
                transcript_string += f"Receiver: {receiver_entry}\n"
        return transcript_string.strip()
    
    def get_context_string(self, transcript:TranscriptConfig):
        params_string = ""
        if self.config.agent_type == "sender" or self.config.agent_type == "receiver" and transcript.params is not None:
            for key, value in transcript.params["public"].items():
                params_string += f"{key}: {value}\n"
        if self.config.agent_type == "sender":
            for key, value in transcript.params["private"].items():
                params_string += f"{key}: {value}\n"
        return params_string.strip()
    
    def get_rounds_statement(self):
        # Fills <ROUNDS_STATEMENT>. total_steps is set by take_turn(); callers that bypass it
        # (rl/ scripts calling construct_messages) get 3.
        n = int(getattr(self, "total_steps", 3) or 3)
        if n == 1:
            return "There is 1 round of interaction."
        return f"There are {n} rounds of interactions."

    def get_behavioral_parameters_string(self, transcript:TranscriptConfig):
        behavioral_parameters_string = ""
        if self.behavioral_parameters is not None:
            for key, value in self.behavioral_parameters.items():
                    behavioral_parameters_string += f"{key}: {value}\n"
        return behavioral_parameters_string.strip()
            
    
    def create_behavioral_parameters_message(self, transcript:TranscriptConfig):
        if transcript.params is None or len(transcript.params) == 0:
            return ""
        else:
            return self.fill_in_content(self.partials["behavioral_parameters"], transcript)
    
    def create_transcript_message(self, transcript:TranscriptConfig):
        if len(transcript.rounds) == 0:
            return ""
        else:
            return self.fill_in_content(self.partials["transcript"], transcript)
        
    def create_context_message(self, transcript:TranscriptConfig):
        if transcript.params is None or len(transcript.params) == 0:
            return ""
        else:
            return self.fill_in_content(self.partials["context"], transcript)
    
    def get_strategy_guide(self):
        # Lazy import: only prompts containing <STRATEGY_GUIDE> need rl.strategy_audit.
        from rl.strategy_audit.taxonomy import render_guide
        return render_guide()

    def fill_in_content(self, content: str, transcript:TranscriptConfig):
        placeholders = {
            "TRANSCRIPT": lambda: self.get_transcript_string(transcript),
            "TRANSCRIPT_MESSAGE": lambda: self.create_transcript_message(transcript),
            "CONTEXT": lambda: self.get_context_string(transcript),
            "CONTEXT_MESSAGE": lambda: self.create_context_message(transcript),
            "BEHAVIORAL_PARAMETERS": lambda: self.get_behavioral_parameters_string(transcript),
            "BEHAVIORAL_PARAMETERS_MESSAGE": lambda: self.create_behavioral_parameters_message(transcript),
            "ROUNDS_STATEMENT": lambda: self.get_rounds_statement(),
            "PRIOR": lambda: str(0.5),
            "STATE": lambda: "guilty",
            "WORD_LIMIT": lambda: str(self.config.word_limit),
            "STRATEGY_GUIDE": lambda: self.get_strategy_guide(),
        }
        for placeholder, placeholder_filler in placeholders.items():
            if f"<{placeholder}>" in content:
                content = content.replace(f"<{placeholder}>", placeholder_filler())
        return content
    
    def construct_messages(self, transcript:TranscriptConfig):
        messages = []
        for message in self.messages:
            messages.append({
                    "role": message["role"],
                    "content": self.fill_in_content(message["content"], transcript),
                })
        return messages
    
    def extract_message(self, response:str):
        if not response:
            return ""
            
        response = response.replace("[argument]", "<argument>")
        response = response.replace("[/argument]", "</argument>")
        
        try:
            parts = response.split("<argument>")
            if len(parts) < 2:
                return ""
            content = parts[1].split("</argument>")[0]
            return content.strip()
        except (IndexError, AttributeError):
            return ""
    
    def extract_action(self, response:str):
        # Last <action> block, not the first (on gpt-oss the first was a stray analysis-channel
        # tag). See parse_action.
        return parse_action(response)
    
    def get_completion(self, transcript:TranscriptConfig):
        prompt = self.construct_messages(transcript)
        responses = self.api_handler(
            model_id=self.config.language_model.model_id,
            prompt=prompt,
        )
        return responses.strip()
    
    def take_turn(self, transcript:TranscriptConfig, current_step: int, total_steps: int = 3):
        self.total_steps = int(total_steps)
        response = self.get_completion(transcript)
        return response