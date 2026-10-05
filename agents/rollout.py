"""Game rollout: the transcript schema + the sequential sender -> receiver round loop."""
from typing import List, Optional, Union

from pydantic import BaseModel


class Round(BaseModel):
    sender: Optional[Union[str, dict]] = None
    receiver: Optional[Union[str, dict]] = None

class TranscriptConfig(BaseModel):
    index: int = 0
    rollout_type: Optional[str] = None
    params: Optional[dict] = {}
    rounds: List[Round] = []
    responses: Optional[List[Round]] = []


class SequentialRollout:
    def __init__(self, sender, receiver, cfg):
        self.sender = sender
        self.receiver = receiver
        self.cfg = cfg

    def game_run(self, transcript: TranscriptConfig, current_step: int):
        new_round = {}
        new_responses = {}
        transcript.rounds.append(Round(**new_round))
        transcript.responses.append(Round(**new_responses))
        new_responses["sender"] = self.sender.take_turn(transcript, current_step, self.cfg.num_steps)
        new_round["sender"] = self.sender.extract_message(new_responses["sender"])
        transcript.rounds[-1] = Round(**new_round)
        transcript.responses[-1] = Round(**new_responses)

        new_responses["receiver"] = self.receiver.take_turn(transcript, current_step, self.cfg.num_steps)
        new_round["receiver"] = self.receiver.extract_action(new_responses["receiver"])
        transcript.rounds[-1] = Round(**new_round)
        transcript.responses[-1] = Round(**new_responses)
        return transcript

    def run(self, game):
        transcript = TranscriptConfig(
            index=game["id"],
            rollout_type="sequential",
            params=game["params"],
        )
        current_step = 0
        while current_step < self.cfg.num_steps:
            transcript = self.game_run(transcript, current_step)
            current_step += 1
        complete = current_step >= self.cfg.num_steps
        return {"transcript": transcript.json(), "complete": complete}
