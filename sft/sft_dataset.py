"""PersuasionMultiTurnSFTDataset: verl's MultiTurnSFTDataset + the Qwen3 thinking-template fix.

MultiTurnSFTDataset tokenizes each message as render(messages[:i+1]) - render(messages[:i]),
assuming each render is a prefix of the next. The stock Qwen3 thinking template (e.g. Qwen3-8B,
checked under transformers 5.6) renders an empty '<think>\\n\\n</think>\\n\\n' stub on an assistant
turn only when it is the last message rendered, so every round-advance user turn loses its
'<|im_start|>user\\n' header (upstream only logs a warning).

The fix renders the stub on every assistant turn. That makes deltas exact and matches GRPO token for
token: the verl rollout (enable_thinking=False, rl/patch_verl.sh 8) emits the same stub in each
assistant generation prompt, where it is loss-masked.

The patch is applied to a deep copy of the tokenizer, because the trainer passes the original to
FSDPCheckpointManager and the saved checkpoint must keep the stock template.
Qwen3-4B-Instruct-2507 (non-thinking, already prefix-consistent) passes through unpatched.

verl loads this via data.custom_cls (file path + class name), so use absolute imports only.
"""
import copy
import math

import numpy as np

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

# Assistant-branch conditional of the stock Qwen3 thinking template (Qwen3-8B
# tokenizer_config.json); asserted present so a template change fails instead of mis-training.
_QWEN3_THINK_BLOCK = """\
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}"""

# every assistant turn gets the think stub (empty when reasoning_content == ''), as in GRPO
_QWEN3_THINK_REPLACEMENT = (
    "        {{- '<|im_start|>' + message.role + '\\n<think>\\n' + "
    "reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}"
)


def patch_qwen3_thinking_template(tokenizer):
    """Return a tokenizer whose chat template renders assistant turns prefix-consistently.

    Non-thinking templates are returned as the same object; thinking templates are patched on a
    deep copy, since the trainer also hands the tokenizer to the checkpoint manager."""
    tpl = tokenizer.chat_template or ""
    if "ns.last_query_index" not in tpl or "<think>" not in tpl:
        return tokenizer
    if _QWEN3_THINK_BLOCK not in tpl:
        raise AssertionError(
            "Qwen3 thinking-style chat template drifted from the pinned assistant-branch block; "
            "sft/sft_dataset.py's prefix-consistency patch does not apply. Re-derive "
            "_QWEN3_THINK_BLOCK from the model's tokenizer_config.json before training.")
    patched = copy.deepcopy(tokenizer)
    patched.chat_template = tpl.replace(_QWEN3_THINK_BLOCK, _QWEN3_THINK_REPLACEMENT)
    return patched


class PersuasionMultiTurnSFTDataset(MultiTurnSFTDataset):
    """MultiTurnSFTDataset over the datasets/old_bailey/sft/build_sft_dataset.py parquets.

    Uses the patched Qwen3 template and repairs `loss_mask` after the parquet round-trip: pyarrow
    returns present values as floats and absent ones as None/NaN, but upstream asserts an int."""

    def __init__(self, parquet_files, tokenizer, config=None, max_samples=-1):
        super().__init__(parquet_files, tokenizer, config=config, max_samples=max_samples)
        self.tokenizer = patch_qwen3_thinking_template(self.tokenizer)
        for messages in self.messages:
            for m in messages:
                if "loss_mask" not in m:
                    continue
                v = m["loss_mask"]
                # seen: None and float; np.floating covers other pyarrow/pandas versions
                if v is None or (isinstance(v, (float, np.floating)) and math.isnan(float(v))):
                    del m["loss_mask"]
                else:
                    m["loss_mask"] = int(v)
