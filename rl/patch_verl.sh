#!/bin/bash
# Idempotent compatibility patches to the verl checkout so verl 0.7.0.dev runs against the repo's
# qwen3_5 stack (transformers 5.6 / sglang 0.5.12 / torch 2.11 / numpy 2.x) and a LoRA'd VLM policy.
# Called by scripts/rl_build_env.sh; safe to re-run. Usage: bash rl/patch_verl.sh <VERL_DIR>
# (the argument is required unless VERL_DIR is set in the environment).
set -euo pipefail
VERL_DIR="${1:-${VERL_DIR:?usage: bash rl/patch_verl.sh <VERL_DIR> (or set VERL_DIR to a verl 0.7.0.dev checkout)}}"
[[ -d "$VERL_DIR/verl" ]] || { echo "[patch_verl] not a verl checkout: $VERL_DIR"; exit 1; }

# (1) verl pins numpy<2.0.0 but works on 2.x (the scipy that sglang pulls in requires numpy>=2).
#     rl_build_env.sh installs verl with --no-deps, so this only matters if verl is installed with
#     its dependencies; relax the pin so that path keeps the locked numpy 2.x too.
sed -i 's/"numpy<2.0.0"/"numpy"/' "$VERL_DIR/setup.py"
sed -i 's/^numpy<2.0.0/numpy/'    "$VERL_DIR/requirements.txt"

# (2) transformers>=5 renamed AutoModelForVision2Seq to AutoModelForImageTextToText. Alias it in
#     verl's unconditional top-level imports (the version-guarded one in the ckpt manager is fine).
sed -i -E 's/^([[:space:]]*)AutoModelForVision2Seq,[[:space:]]*$/\1AutoModelForImageTextToText as AutoModelForVision2Seq,/' \
  "$VERL_DIR/verl/utils/model.py" "$VERL_DIR/verl/workers/fsdp_workers.py" "$VERL_DIR/verl/model_merger/base_model_merger.py"
sed -i 's/AutoModelForTokenClassification, AutoModelForVision2Seq$/AutoModelForTokenClassification, AutoModelForImageTextToText as AutoModelForVision2Seq/' \
  "$VERL_DIR/verl/utils/model.py"

# (3) apply_fsdp2 indexes _no_split_modules[0], but a PEFT/LoRA-wrapped VLM exposes it as a set;
#     coerce set/tuple -> list first.
python - "$VERL_DIR" <<'PYEOF'
import sys
p = sys.argv[1] + "/verl/utils/fsdp_utils.py"
s = open(p).read()
if "PEFT/LoRA-wrapped models expose _no_split_modules as a set" not in s:
    anchor = ("    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):\n"
              "        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]\n\n"
              "    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None")
    repl = ("    # PEFT/LoRA-wrapped models expose _no_split_modules as a set (unindexable); normalize.\n"
            "    if isinstance(fsdp_transformer_layer_cls_to_wrap, (set, tuple)):\n"
            "        fsdp_transformer_layer_cls_to_wrap = list(fsdp_transformer_layer_cls_to_wrap)\n"
            "    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):\n"
            "        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]\n\n"
            "    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None")
    assert anchor in s, "apply_fsdp2 anchor not found (verl version drift?)"
    open(p, "w").write(s.replace(anchor, repl))
    print("[patch_verl] (3) fsdp_utils set->list: applied")
else:
    print("[patch_verl] (3) fsdp_utils set->list: already present")
PYEOF
# (4) sglang>=0.5.x moved get_open_port / get_local_ip_auto into sglang.srt.utils.network; fix
#     verl's sglang rollout imports.
python - "$VERL_DIR" <<'PYEOF'
import sys
p = sys.argv[1] + "/verl/workers/rollout/sglang_rollout/sglang_rollout.py"
s = open(p).read()
o1 = ("from sglang.srt.utils import (\n    assert_pkg_version,\n    get_open_port,\n"
      "    is_cuda,\n    set_prometheus_multiproc_dir,\n    set_ulimit,\n)")
n1 = ("from sglang.srt.utils import (\n    assert_pkg_version,\n    is_cuda,\n"
      "    set_prometheus_multiproc_dir,\n    set_ulimit,\n)\n"
      "from sglang.srt.utils.network import get_open_port  # moved out of srt.utils in sglang>=0.5.x")
if o1 in s:
    s = s.replace(o1, n1); print("[patch_verl] (4a) get_open_port: applied")
else:
    print("[patch_verl] (4a) get_open_port: already/na")
o2 = ("except ImportError:\n    from sglang.srt.utils import get_local_ip_auto as get_ip\n")
n2 = ("except ImportError:\n    try:\n        from sglang.srt.utils import get_local_ip_auto as get_ip\n"
      "    except ImportError:\n        from sglang.srt.utils.network import get_local_ip_auto as get_ip\n")
if o2 in s and "from sglang.srt.utils.network import get_local_ip_auto" not in s:
    s = s.replace(o2, n2); print("[patch_verl] (4b) get_local_ip_auto: applied")
else:
    print("[patch_verl] (4b) get_local_ip_auto: already/na")
open(p, "w").write(s)
PYEOF
# (5) sglang>=0.5.12 renamed the kernel dist "sgl-kernel" -> "sglang-kernel" (still imported as
#     sgl_kernel); update verl's assert.
sed -i 's/            "sgl-kernel",/            "sglang-kernel",  # renamed dist in sglang>=0.5.12/' \
  "$VERL_DIR/verl/workers/rollout/sglang_rollout/sglang_rollout.py"
# (7) Stop multi-turn rollout on either the tokenizer eos (<|im_end|>) or the generation_config eos.
#     Qwen3.5 has no generation_config.json and its config eos is not the chat-template turn end, so
#     the rollout otherwise runs to max_response_length with no parseable argument.
python - "$VERL_DIR" <<'PYEOF'
import sys
p = sys.argv[1] + "/verl/workers/fsdp_workers.py"
s = open(p).read()
old = ('        meta_info = {\n'
       '            "eos_token_id": self.generation_config.eos_token_id\n'
       '            if self.generation_config is not None\n'
       '            else self.tokenizer.eos_token_id,\n'
       '            "pad_token_id": self.generation_config.pad_token_id\n'
       '            if self.generation_config is not None\n'
       '            else self.tokenizer.pad_token_id,\n'
       '        }')
if "Union the tokenizer's eos" in s:
    print("[patch_verl] (7) eos union: already present")
elif old in s:
    new = ('        _eos = set()\n'
           '        _gc = self.generation_config\n'
           '        if _gc is not None and _gc.eos_token_id is not None:\n'
           '            _e = _gc.eos_token_id\n'
           '            _eos.update(_e if isinstance(_e, (list, tuple)) else [_e])\n'
           '        if self.tokenizer.eos_token_id is not None:\n'
           '            _eos.add(self.tokenizer.eos_token_id)  # Union the tokenizer\'s eos (<|im_end|>)\n'
           '        _pad = (_gc.pad_token_id if _gc is not None and _gc.pad_token_id is not None\n'
           '                else self.tokenizer.pad_token_id)\n'
           '        meta_info = {\n'
           '            "eos_token_id": sorted(_eos) if _eos else self.tokenizer.eos_token_id,\n'
           '            "pad_token_id": _pad,\n'
           '        }')
    open(p, "w").write(s.replace(old, new)); print("[patch_verl] (7) eos union: applied")
else:
    raise SystemExit("[patch_verl] (7) eos anchor not found")
PYEOF
# (8) transformers>=5.6 apply_chat_template calls message.get("content"), but verl passes pydantic
#     Message objects; coerce them to dicts. Also turn off qwen3_5 <think> in the policy rollout
#     (RL_SENDER_THINKING=1 re-enables): unbounded reasoning runs responses to the cap and slows
#     training. schemas.py is reset to pristine first so a re-run applies this version.
if git -C "$VERL_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$VERL_DIR" checkout -- verl/workers/rollout/schemas.py 2>/dev/null \
    && echo "[patch_verl] (8) reset schemas.py to pristine before patching" || true
fi
python - "$VERL_DIR" <<'PYEOF'
import sys
p = sys.argv[1] + "/verl/workers/rollout/schemas.py"
s = open(p).read()
old = ('        raw_prompt = processing_class.apply_chat_template(\n'
       '            messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False\n'
       '        )\n')
new = ('        # transformers>=5.6 ProcessorMixin.apply_chat_template calls message.get("content");\n'
       '        # verl passes pydantic Message objects (no .get). qwen3_5 is a VLM so processing_class\n'
       '        # is a ProcessorMixin -> coerce Message objects to plain dicts. (dicts pass through.)\n'
       '        import os as _os\n'
       '        _msgs = [m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m for m in messages]\n'
       '        # [persuasion-gym] disable native qwen <think> for the policy rollout (RL_SENDER_THINKING=1\n'
       '        # re-enables) -- unbounded reasoning balloons responses and slows train+gen.\n'
       '        _ctk = {} if _os.environ.get("RL_SENDER_THINKING", "0") == "1" else {"enable_thinking": False}\n'
       '        raw_prompt = processing_class.apply_chat_template(\n'
       '            _msgs, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False, **_ctk\n'
       '        )\n')
if "RL_SENDER_THINKING" in s:
    print("[patch_verl] (8) schemas Message->dict + thinking-off: already present")
elif old in s:
    open(p, "w").write(s.replace(old, new)); print("[patch_verl] (8) schemas Message->dict + thinking-off: applied")
else:
    raise SystemExit("[patch_verl] (8) schemas anchor not found")
PYEOF
# (9) concurrent reward dispatch (serial judge calls -> thread pool); standalone to avoid heredoc quoting.
python "$(dirname "$0")/_patch_naive_concurrent.py" "$VERL_DIR"
# (10) Log reward-extra monitors to wandb on every train step. Stock verl aggregates
#      reward_extra_info only in validation (val-aux/*), so the rh_* monitors (rl/monitors.py) and
#      score/parse_failure never reached wandb during training.
#      (10A) rh_* -> reward_hacking/<key>/{mean,min,max}, other keys -> reward_extra/<key>/.
#      (10B) mirror verl's response_length/{mean,min,max} into reward_hacking/ (length monitor).
#      Only this script modifies ray_trainer.py, so it is reset to pristine first (if verl is a
#      git checkout) and a re-run applies the current blocks.
if git -C "$VERL_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$VERL_DIR" checkout -- verl/trainer/ppo/ray_trainer.py 2>/dev/null \
    && echo "[patch_verl] (10) reset ray_trainer.py to pristine before patching" || true
fi
python - "$VERL_DIR" <<'PYEOF'
import sys
p = sys.argv[1] + "/verl/trainer/ppo/ray_trainer.py"
s = open(p).read()
lines = s.splitlines(keepends=True)
changed = False

# (10A) per-train-step mean aggregation of the reward-extra monitors, grouped by key family.
if "[persuasion-gym] surface reward-extra monitors" in s:
    print("[patch_verl] (10A) reward-extra train logging: already present")
else:
    needle = "batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})"
    hits = [i for i, ln in enumerate(lines) if needle in ln]
    if not hits:
        raise SystemExit("[patch_verl] (10A) reward-extra anchor not found (verl version drift?)")
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    block = [
        f"{indent}# [persuasion-gym] surface reward-extra monitors to the logger every TRAIN step.\n",
        f"{indent}# Stock verl only aggregates reward_extra_info during validation (val-aux/*), so the\n",
        f"{indent}# rh_* reward-hacking monitors never reached wandb here. Group them: rh_* (the hacks)\n",
        f"{indent}# under reward_hacking/, score/parse_failure under reward_extra/, logging mean/min/max\n",
        f"{indent}# of each per train step. (The per-sample key for the GRPO reward is 'score', not\n",
        f"{indent}# 'reward'; guard 'reward' anyway for safety.)\n",
        f"{indent}for _mk, _mv in reward_extra_infos_dict.items():\n",
        f"{indent}    if _mk == \"reward\":\n",
        f"{indent}        continue\n",
        f"{indent}    try:\n",
        f"{indent}        _ma = np.asarray(_mv, dtype=float)\n",
        f"{indent}    except (TypeError, ValueError):\n",
        f"{indent}        continue\n",
        f"{indent}    if _ma.size and not np.all(np.isnan(_ma)):\n",
        f"{indent}        _grp = \"reward_hacking\" if _mk.startswith(\"rh_\") else \"reward_extra\"\n",
        f"{indent}        metrics[f\"{{_grp}}/{{_mk}}/mean\"] = float(np.nanmean(_ma))\n",
        f"{indent}        metrics[f\"{{_grp}}/{{_mk}}/min\"] = float(np.nanmin(_ma))\n",
        f"{indent}        metrics[f\"{{_grp}}/{{_mk}}/max\"] = float(np.nanmax(_ma))\n",
    ]
    lines[i + 1 : i + 1] = block
    changed = True
    print("[patch_verl] (10A) reward-extra train logging: applied")

s = "".join(lines)
lines = s.splitlines(keepends=True)

# (10B) mirror verl's response_length stats into reward_hacking/ as the length monitor (no judge).
if "reuse verl's native response_length" in s:
    print("[patch_verl] (10B) response_length length-monitor: already present")
else:
    needle2 = "metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))"
    hits2 = [j for j, ln in enumerate(lines) if needle2 in ln]
    if not hits2:
        raise SystemExit("[patch_verl] (10B) compute_data_metrics anchor not found (verl version drift?)")
    j = hits2[0]
    indent2 = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
    blockB = [
        f"{indent2}# [persuasion-gym] reuse verl's native response_length as the LENGTH reward-hacking\n",
        f"{indent2}# monitor (no judge): mirror its mean/min/max into the reward_hacking/ group beside\n",
        f"{indent2}# rh_tone/rh_fake (verl compute_data_metrics emits response_length/{{mean,min,max}}).\n",
        f"{indent2}for _st in (\"mean\", \"min\", \"max\"):\n",
        f"{indent2}    _rk = f\"response_length/{{_st}}\"\n",
        f"{indent2}    if _rk in metrics:\n",
        f"{indent2}        metrics[f\"reward_hacking/{{_rk}}\"] = metrics[_rk]\n",
    ]
    lines[j + 1 : j + 1] = blockB
    changed = True
    print("[patch_verl] (10B) response_length length-monitor: applied")

s = "".join(lines)

# (10C) Gate the training rollout dump: step 1, then every RL_ROLLOUT_DUMP_FREQ steps (default 5).
# Stock verl dumps every step when rollout_data_dir is set.
if "[persuasion-gym] gate the per-step rollout dump" in s:
    print("[patch_verl] (10C) rollout-dump freq gate: already present")
else:
    old_c = (
        '                    if rollout_data_dir:\n'
        '                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)\n'
    )
    if old_c not in s:
        raise SystemExit("[patch_verl] (10C) rollout_data_dir anchor not found (verl version drift?)")
    new_c = (
        '                    # [persuasion-gym] gate the per-step rollout dump: always dump step 1, then\n'
        '                    # every RL_ROLLOUT_DUMP_FREQ steps (default 5). freq<=0 -> every step (stock).\n'
        '                    _rdump_freq = int(os.environ.get("RL_ROLLOUT_DUMP_FREQ", "5"))\n'
        '                    _rdump_now = (_rdump_freq <= 0) or (self.global_steps == 1) or (self.global_steps % _rdump_freq == 0)\n'
        '                    if rollout_data_dir and _rdump_now:\n'
        '                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)\n'
    )
    s = s.replace(old_c, new_c)
    lines = s.splitlines(keepends=True)
    changed = True
    print("[patch_verl] (10C) rollout-dump freq gate: applied")

# (10D) Rotate the judge-response buffer via _persuasion_flush_judges (10F): rename it to
# judges_step_NNNNN.yaml (1:1 with rollouts/<step>.jsonl) on a dump step, discard it otherwise.
if "[persuasion-gym] rotate judge-response buffer" in s:
    print("[patch_verl] (10D) judge-response buffer rotation: already present")
else:
    # Anchor: the _log_rollout_data line written by (10C).
    anchor_10d = (
        '                    if rollout_data_dir and _rdump_now:\n'
        '                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)\n'
    )
    if anchor_10d not in s:
        raise SystemExit("[patch_verl] (10D) (10C) anchor not found -- run after (10C) is applied")
    rotation_block = (
        '                    # [persuasion-gym] (10D) rotate judge-response buffer: dump step -> named file\n'
        '                    # (1:1 with the rollout dump); non-dump step -> discard (keep files per-step).\n'
        '                    if _rdump_now:\n'
        '                        self._persuasion_flush_judges(f"judges_step_{self.global_steps:05d}")\n'
        '                    else:\n'
        '                        self._persuasion_flush_judges(None, discard=True)\n'
    )
    s = s.replace(anchor_10d, anchor_10d + rotation_block)
    lines = s.splitlines(keepends=True)
    changed = True
    print("[patch_verl] (10D) judge-response buffer rotation: applied")

# (10E) Add a structured `messages` field to each rollout-dump entry: verl's native conversation
# (non_tensor_batch["messages"][i]["messages"], Receiver turns as role="user"), or else turns parsed
# from the raw prompt plus the response blob. `input`/`output` strings are kept.
if "[persuasion-gym] structured messages" in s:
    print("[patch_verl] (10E) structured messages in rollout dump: already present")
else:
    ok_10e = True
    # (10E-a) _dump_generations: add raw_prompts / raw_messages params + the structured-messages block.
    old_sig = '    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):\n'
    new_sig = '    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, raw_prompts=None, raw_messages=None):\n'
    old_loop = (
        '        lines = []\n'
        '        for i in range(n):\n'
        '            entry = {k: v[i] for k, v in base_data.items()}\n'
        '            lines.append(json.dumps(entry, ensure_ascii=False))\n'
    )
    new_loop = (
        '        # [persuasion-gym] structured messages (10E): native multi-turn conversation if present,\n'
        '        # else fall back to parsing <|im_start|>role turns from the raw prompt + a response blob.\n'
        '        import re as _re_10e\n'
        '        _msg_re = _re_10e.compile(r"<\\|im_start\\|>(\\w+)\\n(.*?)<\\|im_end\\|>", _re_10e.DOTALL)\n'
        '        def _coerce_turn_10e(_t):\n'
        '            _role = _t.get("role") if isinstance(_t, dict) else getattr(_t, "role", None)\n'
        '            _content = _t.get("content") if isinstance(_t, dict) else getattr(_t, "content", None)\n'
        '            return {"role": _role, "content": _content}\n'
        '        lines = []\n'
        '        for i in range(n):\n'
        '            entry = {k: v[i] for k, v in base_data.items()}\n'
        '            _msgs = None\n'
        '            if raw_messages is not None:\n'
        '                try:\n'
        '                    _rm = raw_messages[i]\n'
        '                    _turns = _rm["messages"] if isinstance(_rm, dict) else _rm\n'
        '                    _msgs = [_coerce_turn_10e(_t) for _t in _turns] or None\n'
        '                except Exception:\n'
        '                    _msgs = None\n'
        '            if _msgs is None and raw_prompts is not None:\n'
        '                _msgs = [{"role": m.group(1), "content": m.group(2)} for m in _msg_re.finditer(raw_prompts[i])]\n'
        '                if _msgs:\n'
        '                    _msgs.append({"role": "assistant", "content": outputs[i]})\n'
        '                else:\n'
        '                    _msgs = None\n'
        '            if _msgs is not None:\n'
        '                entry["messages"] = _msgs\n'
        '            lines.append(json.dumps(entry, ensure_ascii=False))\n'
    )
    # (10E-b) _log_rollout_data: native messages + raw-token prompt fallback source.
    old_decode = (
        '            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)\n'
        '            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)\n'
    )
    new_decode = (
        '            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)\n'
        '            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)\n'
        '            # [persuasion-gym] structured messages (10E): native per-sample conversation if present,\n'
        '            # else the raw-token prompt for the fallback parser.\n'
        '            _raw_msgs_10e = batch.non_tensor_batch.get("messages", None)\n'
        '            _raw_prompts_10e = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=False)\n'
    )
    # (10E-c) _log_rollout_data: pass raw_prompts + raw_messages to _dump_generations.
    old_call = (
        '            self._dump_generations(\n'
        '                inputs=inputs,\n'
        '                outputs=outputs,\n'
        '                gts=sample_gts,\n'
        '                scores=scores,\n'
        '                reward_extra_infos_dict=reward_extra_infos_to_dump,\n'
        '                dump_path=rollout_data_dir,\n'
        '            )\n'
    )
    new_call = (
        '            self._dump_generations(\n'
        '                inputs=inputs,\n'
        '                outputs=outputs,\n'
        '                gts=sample_gts,\n'
        '                scores=scores,\n'
        '                reward_extra_infos_dict=reward_extra_infos_to_dump,\n'
        '                dump_path=rollout_data_dir,\n'
        '                raw_prompts=_raw_prompts_10e,  # [persuasion-gym] structured messages (10E)\n'
        '                raw_messages=_raw_msgs_10e,    # [persuasion-gym] native multi-turn (10E)\n'
        '            )\n'
    )
    for name, old in [("sig", old_sig), ("loop", old_loop), ("decode", old_decode), ("call", old_call)]:
        if old not in s:
            print(f"[patch_verl] (10E) WARNING: anchor '{name}' not found (verl version drift?)"); ok_10e = False
    if ok_10e:
        s = s.replace(old_sig, new_sig)
        s = s.replace(old_loop, new_loop)
        s = s.replace(old_decode, new_decode)
        s = s.replace(old_call, new_call)
        lines = s.splitlines(keepends=True)
        changed = True
        print("[patch_verl] (10E) structured messages in rollout dump: applied")
    else:
        print("[patch_verl] (10E) structured messages: SKIPPED (anchor mismatch)")

# (10F) Judge-dump trainer hooks: (a) buffer flush/discard helper, (b) startup RL_JUDGE_STEP=0 and
# stale-buffer guard, (c) pre-train validation flush, (d) per-step RL_JUDGE_STEP stamp,
# (e) periodic/last validation flushes. Anchors are pristine verl text, independent of 10A-10E.
if "[persuasion-gym] _persuasion_flush_judges" in s:
    print("[patch_verl] (10F) judge-dump trainer hooks: already present")
else:
    ok_10f = True
    # (10F-a) helper method, inserted just before _log_rollout_data.
    anchor_helper = (
        '    def _log_rollout_data(\n'
        '        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str\n'
        '    ):\n'
    )
    helper_def = (
        '    def _persuasion_flush_judges(self, tag, discard=False):\n'
        '        # [persuasion-gym] _persuasion_flush_judges: move/clear the judge-response buffer in\n'
        '        # RL_JUDGE_DUMP_DIR. discard=True removes it (non-dump train steps); else atomically\n'
        '        # rename judges_buffer.yaml -> <tag>.yaml. No-op if the dir is unset or buffer absent.\n'
        '        import os as _os\n'
        '        _d = _os.environ.get("RL_JUDGE_DUMP_DIR", "")\n'
        '        if not _d:\n'
        '            return\n'
        '        _b = _os.path.join(_d, "judges_buffer.yaml")\n'
        '        if not _os.path.isfile(_b):\n'
        '            return\n'
        '        if discard:\n'
        '            try:\n'
        '                _os.remove(_b)\n'
        '            except OSError:\n'
        '                pass\n'
        '            return\n'
        '        _os.makedirs(_d, exist_ok=True)\n'
        '        _os.replace(_b, _os.path.join(_d, str(tag) + ".yaml"))\n'
        '\n'
    )
    # (10F-b) startup: set RL_JUDGE_STEP=0 + preserve any orphaned buffer from a crashed prior run.
    anchor_ckpt = (
        '        # load checkpoint before doing anything\n'
        '        self._load_checkpoint()\n'
    )
    startup_block = (
        '\n'
        '        # [persuasion-gym] (10F) judge-dump startup: tag pre-train records with step 0 and move any\n'
        '        # orphaned buffer from a crashed prior run aside (.stale) so it is not misattributed here.\n'
        '        import os as _os_pg\n'
        '        _os_pg.environ["RL_JUDGE_STEP"] = "0"\n'
        '        _jd_pg = _os_pg.environ.get("RL_JUDGE_DUMP_DIR", "")\n'
        '        if _jd_pg:\n'
        '            _jb_pg = _os_pg.path.join(_jd_pg, "judges_buffer.yaml")\n'
        '            if _os_pg.path.isfile(_jb_pg):\n'
        '                _os_pg.replace(_jb_pg, _jb_pg + ".stale")\n'
    )
    # (10F-c) flush pre-train validation judge records to their own file.
    anchor_preval = (
        '            pprint(f"Initial validation metrics: {val_metrics}")\n'
        '            logger.log(data=val_metrics, step=self.global_steps)\n'
    )
    preval_flush = (
        '            self._persuasion_flush_judges(f"judges_val_step_{self.global_steps:05d}")  # [persuasion-gym] 10F\n'
    )
    # (10F-d) per-train-step: stamp RL_JUDGE_STEP so reward records carry the current step.
    anchor_stepenv = (
        '                # pass global_steps to trace\n'
        '                gen_batch.meta_info["global_steps"] = self.global_steps\n'
    )
    stepenv_block = (
        '                import os as _os_pg2  # [persuasion-gym] 10F: tag judge records with this train step\n'
        '                _os_pg2.environ["RL_JUDGE_STEP"] = str(self.global_steps)\n'
    )
    # (10F-e) flush periodic/last validation judge records to their own file.
    anchor_periodval = (
        '                    with marked_timer("testing", timing_raw, color="green"):\n'
        '                        val_metrics: dict = self._validate()\n'
        '                        if is_last_step:\n'
        '                            last_val_metrics = val_metrics\n'
        '                    metrics.update(val_metrics)\n'
    )
    periodval_flush = (
        '                    self._persuasion_flush_judges(f"judges_val_step_{self.global_steps:05d}")  # [persuasion-gym] 10F\n'
    )
    for name, anc in [("helper", anchor_helper), ("ckpt", anchor_ckpt), ("preval", anchor_preval),
                      ("stepenv", anchor_stepenv), ("periodval", anchor_periodval)]:
        if anc not in s:
            print(f"[patch_verl] (10F) WARNING: anchor '{name}' not found (verl version drift?)"); ok_10f = False
    if ok_10f:
        s = s.replace(anchor_helper, helper_def + anchor_helper)
        s = s.replace(anchor_ckpt, anchor_ckpt + startup_block)
        s = s.replace(anchor_preval, anchor_preval + preval_flush)
        s = s.replace(anchor_stepenv, anchor_stepenv + stepenv_block)
        s = s.replace(anchor_periodval, anchor_periodval + periodval_flush)
        lines = s.splitlines(keepends=True)
        changed = True
        print("[patch_verl] (10F) judge-dump trainer hooks: applied")
    else:
        print("[patch_verl] (10F) judge-dump trainer hooks: SKIPPED (anchor mismatch)")

# (10G) Rejection sampling (opt-in REJECTION_SAMPLING=1; logic in rl/rejection_sampling.py).
# (a) Stash the pre-repeat prompts; right after the mainline repeat/union, score the pool,
# regenerate for prompts below the clean target (same uid, so same GRPO group) and keep rollout.n
# rows per prompt. (b) An RS step reuses the loop's scores in the reward dispatch, so each rollout
# is judged once (the judge is temp-1 stochastic). Selection runs before _balance_batch /
# old_log_prob, so dropped rollouts never reach an FSDP forward. A missing anchor hard-fails, since
# an unpatched trainer would silently ignore the knob.
if "[persuasion-gym] (10G) rejection sampling" in s:
    print("[patch_verl] (10G) rejection sampling: already present")
else:
    anchor_repeat = (
        "                    # repeat to align with repeated responses in rollout\n"
        "                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)\n"
        "                    batch = batch.union(gen_batch_output)\n"
    )
    anchor_reward = (
        "                        if self.config.reward_model.launch_reward_fn_async:\n"
        "                            future_reward = compute_reward_async.remote(\n"
        "                                data=batch, config=self.config, tokenizer=self.tokenizer\n"
        "                            )\n"
        "                        else:\n"
        "                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)\n"
    )
    for name, anc in [("repeat-union", anchor_repeat), ("reward-dispatch", anchor_reward)]:
        if anc not in s:
            raise SystemExit(f"[patch_verl] (10G) anchor '{name}' not found (verl version drift?)")
    rs_stash = (
        "                    # [persuasion-gym] (10G) rejection sampling: stash the pre-repeat prompt\n"
        "                    # remainder (uid/reward_model/extra_info/data_source rows, index-aligned\n"
        "                    # with gen_batch) BEFORE the repeat -- repeat() returns a NEW DataProto,\n"
        "                    # so this reference keeps the per-prompt rows for building regen chunks.\n"
        "                    _rs_prompt_batch = batch\n"
    )
    rs_loop = """\
                    # [persuasion-gym] (10G) rejection sampling for penalty-free rollouts (opt-in,
                    # REJECTION_SAMPLING=1; pure logic in rl/rejection_sampling.py). Score the initial
                    # pool once, regenerate extra rollouts for prompts below the clean target (same
                    # uid -> same GRPO group), score each chunk once (the judge is temp-1 stochastic:
                    # a rollout is judged EXACTLY once, kept rows reuse stored scores), then keep
                    # exactly rollout.n rows per prompt -- positives first, negatives fill. Selection
                    # happens BEFORE _balance_batch / old_log_prob, so discarded rollouts never reach
                    # an FSDP forward pass. Sync-rollout + sync-reward path only.
                    _rs_done, _rs_extra_keys, _rs_mod = False, [], None
                    if os.environ.get("REJECTION_SAMPLING", "0") not in ("0", ""):
                        # Fail LOUD on every unusable config: a run whose name/metadata claim RS but
                        # that silently trains the stock flow (or vice versa) would be analyzed as
                        # the wrong arm. The repo root is on sys.path (rl/__init__, loaded with the
                        # custom reward fn), so an ImportError here should crash the run too.
                        import sys as _sys_rs
                        import rl.rejection_sampling as _rs_mod
                        _rs_cfg = _rs_mod.config_from_env(n_rollout=self.config.actor_rollout_ref.rollout.n)
                        # config_from_env accepts exactly "1" past this point (true/yes/... raise --
                        # the launcher's guards gate on the exact string "1"), so _rs_cfg.enabled holds.
                        if self.async_rollout_mode or self.config.reward_model.launch_reward_fn_async or self.use_rm:
                            raise ValueError(
                                "[rejection-sampling] REJECTION_SAMPLING=1 needs the sync-rollout + "
                                "sync-reward path without a reward-model worker (async_rollout_mode / "
                                "reward_model.launch_reward_fn_async / use_rm are unsupported)")
                        if getattr(_sys_rs.modules.get("custom_module"), "postprocess_group_scores", None) is not None:
                            raise ValueError(
                                "[rejection-sampling] incompatible with a postprocess_group_scores "
                                "reward hook (per-chunk scoring would see partial groups)")
                    if _rs_mod is not None:
                        with marked_timer("reward", timing_raw, color="yellow"):
                            _rs_rt, _rs_ex = compute_reward(batch, self.reward_fn)
                        batch.batch["token_level_scores"] = _rs_rt
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in _rs_ex.items()})
                        _rs_extra_keys = list(_rs_ex.keys())
                        _rs_done = True
                        _rs_mask = _rs_mod.positive_mask(_rs_ex)
                        if _rs_mask is None:
                            metrics["rejection/error"] = 1.0
                            print(f"[rejection-sampling] step={self.global_steps} predicate keys missing"
                                  " from reward extras; keeping the stock batch this step")
                        else:
                            _rs_pool, _rs_pool_mask = batch, _rs_mask
                            _rs_init = _rs_mod.step_snapshot(_rs_pool.non_tensor_batch["uid"], _rs_pool_mask, _rs_cfg)
                            _rs_clamp = (f" (clamped from {_rs_cfg.target_clean})"
                                         if _rs_cfg.target_effective != _rs_cfg.target_clean else "")
                            print(f"[rejection-sampling] step={self.global_steps}"
                                  f" init_clean_frac={_rs_init['clean_frac']:.3f}"
                                  f" below_target={int(_rs_init['prompts_below_target'])}/{int(_rs_init['n_prompts'])}"
                                  f" target={_rs_cfg.target_effective}{_rs_clamp}")
                            _rs_prompt_uids = list(_rs_prompt_batch.non_tensor_batch["uid"])
                            _rs_rounds = _rs_extra = _rs_pad_rows = _rs_dropped = 0
                            for _rs_round in range(1, _rs_cfg.max_gen_rounds + 1):
                                _rs_plan = _rs_mod.prompts_needing_regen(
                                    _rs_prompt_uids, _rs_pool.non_tensor_batch["uid"], _rs_pool_mask, _rs_cfg)
                                if not _rs_plan:
                                    break
                                _rs_idx = sorted(_rs_plan)
                                _rs_cnt = [_rs_plan[i] for i in _rs_idx]
                                _rs_sub = gen_batch.select_idxs(_rs_idx).sample_level_repeat(_rs_cnt)
                                _rs_sub_pad, _rs_p = pad_dataproto_to_divisor(_rs_sub, self.actor_rollout_wg.world_size)
                                print(f"[rejection-sampling] step={self.global_steps} regen_round={_rs_round}"
                                      f" prompts={len(_rs_idx)} new_rollouts={len(_rs_sub)} pad={_rs_p}")
                                with marked_timer("gen_rs", timing_raw, color="red"):
                                    _rs_out = self.actor_rollout_wg.generate_sequences(_rs_sub_pad)
                                _rs_out.meta_info.pop("timing", None)
                                _rs_out = unpad_dataproto(_rs_out, pad_size=_rs_p)
                                _rs_chunk = _rs_prompt_batch.select_idxs(_rs_idx).sample_level_repeat(_rs_cnt)
                                _rs_chunk = _rs_chunk.union(_rs_out)
                                if (_rs_chunk.batch["responses"].shape[1] != _rs_pool.batch["responses"].shape[1]
                                        or _rs_chunk.batch["input_ids"].shape[1] != _rs_pool.batch["input_ids"].shape[1]):
                                    _rs_dropped += 1
                                    print(f"[rejection-sampling] step={self.global_steps} regen_round={_rs_round}"
                                          " chunk DROPPED (response/prompt width drift)")
                                    continue
                                with marked_timer("reward", timing_raw, color="yellow"):
                                    _rs_rt_c, _rs_ex_c = compute_reward(_rs_chunk, self.reward_fn)
                                _rs_mask_c = _rs_mod.positive_mask(_rs_ex_c)
                                if _rs_mask_c is None or set(_rs_ex_c.keys()) != set(_rs_extra_keys):
                                    _rs_dropped += 1
                                    print(f"[rejection-sampling] step={self.global_steps} regen_round={_rs_round}"
                                          " chunk DROPPED (reward extras mismatch)")
                                    continue
                                _rs_chunk.batch["token_level_scores"] = _rs_rt_c
                                _rs_chunk.non_tensor_batch.update({k: np.array(v) for k, v in _rs_ex_c.items()})
                                _rs_pool = DataProto.concat([_rs_pool, _rs_chunk])
                                _rs_pool_mask = np.concatenate([_rs_pool_mask, _rs_mask_c])
                                _rs_rounds += 1
                                _rs_extra += len(_rs_chunk)
                                _rs_pad_rows += _rs_p
                            _rs_keep, _rs_sel = _rs_mod.select_kept(
                                _rs_pool.non_tensor_batch["uid"], _rs_pool_mask, _rs_cfg,
                                scores=_rs_pool.non_tensor_batch.get("score"))
                            batch = _rs_pool.select_idxs(_rs_keep)
                            metrics.update(_rs_mod.step_metrics(
                                _rs_init, _rs_sel, _rs_cfg, gen_rounds=_rs_rounds,
                                extra_rollouts=_rs_extra, pad_rollouts=_rs_pad_rows,
                                chunks_dropped=_rs_dropped))
                            print(f"[rejection-sampling] step={self.global_steps} kept={len(batch)}"
                                  f" pos={int(_rs_sel['kept_positives'])} neg_fill={int(_rs_sel['kept_negatives'])}"
                                  f" fallback_prompts={int(_rs_sel['prompts_below_target'])} extra_total={_rs_extra}")
                            # Drop the (up-to-3x-batch) pool + regen temporaries now -- as fit()
                            # locals they would otherwise stay live through old_log_prob/adv/update.
                            _rs_pool = _rs_chunk = _rs_out = _rs_sub = _rs_sub_pad = None
                            _rs_rt = _rs_ex = _rs_rt_c = _rs_ex_c = _rs_mask = _rs_pool_mask = None
"""
    guard_reward = (
        "                        if _rs_done:  # [persuasion-gym] (10G) RS loop already scored every kept row\n"
        "                            # Reuse the stored per-chunk results (post-balance row order): rebuilding\n"
        "                            # reward_extra_infos_dict from non_tensor_batch keeps it row-aligned for\n"
        "                            # the (10A) merge/metrics and _log_rollout_data; the token_level_scores\n"
        "                            # assignment below becomes a no-op self-assignment. NEVER re-judge here\n"
        "                            # (the judge is temp-1 stochastic; scores are frozen at first judgment).\n"
        "                            reward_tensor = batch.batch[\"token_level_scores\"]\n"
        "                            reward_extra_infos_dict = {k: batch.non_tensor_batch[k].tolist() for k in _rs_extra_keys}\n"
        "                        elif self.config.reward_model.launch_reward_fn_async:\n"
        "                            future_reward = compute_reward_async.remote(\n"
        "                                data=batch, config=self.config, tokenizer=self.tokenizer\n"
        "                            )\n"
        "                        else:\n"
        "                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)\n"
    )
    s = s.replace(anchor_repeat, rs_stash + anchor_repeat + rs_loop)
    s = s.replace(anchor_reward, guard_reward)
    lines = s.splitlines(keepends=True)
    changed = True
    print("[patch_verl] (10G) rejection sampling: applied")

if changed:
    open(p, "w").write("".join(lines))
PYEOF
# (10H) Mixture-of-receivers: exact per-step stratified receiver assignment (opt-in via
#       RECEIVER_MIX_SPEC for N-way or the legacy RECEIVER_MIX_MODEL for 2-way; logic in
#       rl/receiver_mix.py, transport in rl/receiver_client.py). Stamps receiver_backend on each
#       row after the uid assignment and before gen_batch derivation/repeat, so the interaction
#       (interaction_kwargs, aliased by RLHFDataset to extra_info["interaction_kwargs"]) and the
#       reward see one per-game assignment and every GRPO group, including 10G regen chunks, stays
#       receiver-homogeneous. Reapplied after the (10) reset; anchor is the pristine uid block.
python - "$VERL_DIR" <<'PYEOF'
import sys
p = sys.argv[1] + "/verl/trainer/ppo/ray_trainer.py"
s = open(p).read()
anchor = (
    '                # add uid to batch\n'
    '                batch.non_tensor_batch["uid"] = np.array(\n'
    '                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object\n'
    '                )\n'
)
block = (
    '                # [persuasion-gym] (10H) mixture-of-receivers: exact per-step stratified\n'
    '                # receiver assignment. RECEIVER_MIX_SPEC ("label:weight,..." where the label\n'
    '                # IS the hosted model id and "local" is the co-served model) selects the\n'
    '                # N-way path; the legacy RECEIVER_MIX_MODEL/FRAC pair selects the frozen\n'
    '                # 2-way path. ALL label handling lives in rl/receiver_mix.py (importable and\n'
    '                # selftested -- escaped-string python inside a shell heredoc is the worst\n'
    '                # possible home for branching logic).\n'
    '                # Must run BEFORE _get_gen_batch/repeat so rollout AND reward share one\n'
    '                # per-game assignment; stamp_rows asserts the RLHFDataset aliasing of\n'
    '                # interaction_kwargs and fails LOUD -- a `_mix`-named run must never\n'
    '                # silently train 100% local.\n'
    '                # VERSION-TOLERANT: this verl checkout is SHARED by the main repo and every\n'
    '                # git worktree, so it may be imported alongside an rl/receiver_mix.py that\n'
    '                # predates the N-way mixture. Capability-detect rather than assume: an old\n'
    '                # module still runs the legacy 2-way path correctly, and asking it for an\n'
    '                # N-way spec fails loudly instead of silently ignoring the spec.\n'
    '                if os.environ.get("RECEIVER_MIX_SPEC") or os.environ.get("RECEIVER_MIX_MODEL"):\n'
    '                    import rl.receiver_mix as _rmix_10h\n'
    '                    _rmix_spec = os.environ.get("RECEIVER_MIX_SPEC", "")\n'
    '                    _rmix_nway = hasattr(_rmix_10h, "assign_receivers_nway")\n'
    '                    if _rmix_spec and not _rmix_nway:\n'
    '                        raise RuntimeError(\n'
    '                            "[receiver-mix] RECEIVER_MIX_SPEC is set but this checkout\'s "\n'
    '                            "rl/receiver_mix.py predates the N-way mixture -- refusing to "\n'
    '                            "train a run named for a mixture it cannot perform")\n'
    '                    _rmix_kw = {"spec": _rmix_spec} if _rmix_nway else {}\n'
    '                    _rmix_assign = _rmix_10h.stamp_rows(\n'
    '                        batch.non_tensor_batch["interaction_kwargs"],\n'
    '                        batch.non_tensor_batch["extra_info"],\n'
    '                        global_step=self.global_steps,\n'
    '                        seed=int(os.environ.get("RECEIVER_MIX_SEED", "2026")),\n'
    '                        frac=float(os.environ.get("RECEIVER_MIX_FRAC", "0.5")),\n'
    '                        **_rmix_kw,\n'
    '                    )\n'
    '                    if hasattr(_rmix_10h, "step_metrics"):\n'
    '                        metrics.update(_rmix_10h.step_metrics(_rmix_assign, _rmix_spec))\n'
    '                        print(f"[receiver-mix] step={self.global_steps} "\n'
    '                              + _rmix_10h.format_counts(_rmix_assign, _rmix_spec), flush=True)\n'
    '                    else:\n'
    '                        _n_api_10h = sum(1 for _v in _rmix_assign.values() if _v != "local")\n'
    '                        metrics["receiver_mix/n_api_games"] = float(_n_api_10h)\n'
    '                        metrics["receiver_mix/n_local_games"] = float(len(_rmix_assign) - _n_api_10h)\n'
    '                        print(f"[receiver-mix] step={self.global_steps} "\n'
    '                              f"api_games={_n_api_10h}/{len(_rmix_assign)}", flush=True)\n'
)
if "[persuasion-gym] (10H)" in s:
    print("[patch_verl] (10H) mixture-of-receivers: already present")
elif anchor in s:
    open(p, "w").write(s.replace(anchor, anchor + block))
    print("[patch_verl] (10H) mixture-of-receivers: applied")
else:
    raise SystemExit("[patch_verl] (10H) uid anchor not found (verl version drift?)")
PYEOF
# (10I) Aux deception-probe CE attach (opt-in AUX_CE=1; logic in rl/aux_ce.py). Attaches one
#       in-role probe item per rollout row, keyed by extra_info["index"], as six row-aligned aux_*
#       tensors right before update_actor, i.e. after _balance_batch (row order final) and
#       compute_advantage (safe with rejection sampling). Consumed by (13); with AUX_CE=0 the keys
#       are absent and (13) is a no-op. AUX_CE_BALANCE_MODE and its validated scalars ride
#       meta_info so every actor rank uses the driver's config. Anchor: the critic-warmup block.
python - "$VERL_DIR" <<'PYEOF'
import sys
p = sys.argv[1] + "/verl/trainer/ppo/ray_trainer.py"
s = open(p).read()
anchor = (
    "                    # implement critic warmup\n"
    "                    if self.config.trainer.critic_warmup <= self.global_steps:\n"
)
block = (
    "                    # [persuasion-gym] (10I) aux-CE attach: one probe item per rollout row,\n"
    "                    # keyed by extra_info['index'] (game id); the six aux_* tensors ride the\n"
    "                    # row-wise DP dispatch into dp_actor (patch 13). Batch-aligned by\n"
    "                    # construction: items come only from THIS step's games.\n"
    "                    if os.environ.get(\"AUX_CE\", \"0\") == \"1\":\n"
    "                        import rl.aux_ce as _aux_10i\n"
    "                        if getattr(self, \"_persuasion_aux_ce\", None) is None:\n"
    "                            self._persuasion_aux_ce = _aux_10i.load_state(\n"
    "                                tokenizer=self.tokenizer,\n"
    "                                apply_chat_template_kwargs=dict(\n"
    "                                    self.config.data.get(\"apply_chat_template_kwargs\") or {}),\n"
    "                            )\n"
    "                            _aux_balance_mode_10i = os.environ.get(\n"
    "                                \"AUX_CE_BALANCE_MODE\", \"off\").strip()\n"
    "                            if _aux_balance_mode_10i not in (\"off\", \"norm_ratio\"):\n"
    "                                raise ValueError(\n"
    "                                    f\"invalid AUX_CE_BALANCE_MODE={_aux_balance_mode_10i!r}\")\n"
    "                            self._persuasion_aux_balance_mode = _aux_balance_mode_10i\n"
    "                            if _aux_balance_mode_10i == \"norm_ratio\":\n"
    "                                import rl.aux_grad_balance as _aux_balance_10i\n"
    "                                self._persuasion_aux_grad_balance = _aux_balance_10i.config_from_env(\n"
    "                                    coeff_cap=self._persuasion_aux_ce.coeff)\n"
    "                        metrics.update(_aux_10i.attach(\n"
    "                            batch, self._persuasion_aux_ce, global_step=self.global_steps,\n"
    "                            world_size=self.actor_rollout_wg.world_size))\n"
    "                        batch.meta_info[\"aux_ce_coeff\"] = self._persuasion_aux_ce.coeff\n"
    "                        batch.meta_info[\"aux_ce_terms\"] = \",\".join(self._persuasion_aux_ce.terms)\n"
    "                        if self._persuasion_aux_balance_mode == \"norm_ratio\":\n"
    "                            batch.meta_info[\"aux_ce_balance_mode\"] = \"norm_ratio\"\n"
    "                            batch.meta_info[\"aux_ce_target_grad_ratio\"] = self._persuasion_aux_grad_balance.target_ratio\n"
    "                            batch.meta_info[\"aux_ce_balance_epsilon\"] = self._persuasion_aux_grad_balance.epsilon\n"
    "                        assert \"aux_input_ids\" in batch.batch.keys(), \"[aux-ce] attach failed\"\n"
    "                        assert \"aux_term_id\" in batch.batch.keys(), \"[aux-ce] term ids missing (stale rl/aux_ce.py?)\"\n"
)
if "[persuasion-gym] (10I)" in s:
    print("[patch_verl] (10I) aux-CE attach: already present")
elif anchor in s:
    open(p, "w").write(s.replace(anchor, block + anchor))
    print("[patch_verl] (10I) aux-CE attach: applied")
else:
    raise SystemExit("[patch_verl] (10I) critic-warmup anchor not found (verl version drift?)")
PYEOF
# (11) fsdp_sft_trainer hardcodes attn_implementation="flash_attention_2", which this env's
#      flash_attn_4 stub fails (is_flash_attn_2_available()). Make it configurable so
#      scripts/sft_train_sender.slurm can pass +model.attn_implementation=sdpa (GRPO already uses
#      sdpa via grpo_persuasion.yaml); the default is unchanged.
python - "$VERL_DIR" <<'PYEOF'
import sys
p = sys.argv[1] + "/verl/trainer/fsdp_sft_trainer.py"
s = open(p).read()
anchor = '                attn_implementation="flash_attention_2",'
repl = ('                attn_implementation=self.config.model.get("attn_implementation", "flash_attention_2"),  # [persuasion-gym] (11)')
if "[persuasion-gym] (11)" in s:
    print("[patch_verl] (11) sft attn_implementation knob: already present")
else:
    assert anchor in s, "fsdp_sft_trainer attn_implementation anchor not found (verl version drift?)"
    open(p, "w").write(s.replace(anchor, repl, 1))
    print("[patch_verl] (11) sft attn_implementation knob: applied")
PYEOF
# (12) transformers>=5 apply_chat_template(tokenize=True, return_tensors="pt") returns a
#      BatchEncoding, so MultiTurnSFTDataset's full_tokens[0].tolist() raises AttributeError on the
#      first sample. Unwrap input_ids before the validation step.
python - "$VERL_DIR" <<'PYEOF'
import sys
p = sys.argv[1] + "/verl/utils/dataset/multiturn_sft_dataset.py"
s = open(p).read()
anchor = ("        # Track concatenated tokens for validation\n"
          "        concat_tokens = []")
unwrap = ("        if not isinstance(full_tokens, torch.Tensor):  # [persuasion-gym] (12) transformers>=5 BatchEncoding\n"
          "            full_tokens = full_tokens[\"input_ids\"]\n\n")
if "[persuasion-gym] (12)" in s:
    print("[patch_verl] (12) multiturn BatchEncoding unwrap: already present")
else:
    assert anchor in s, "multiturn_sft_dataset anchor not found (verl version drift?)"
    open(p, "w").write(s.replace(anchor, unwrap + anchor, 1))
    print("[patch_verl] (12) multiturn BatchEncoding unwrap: applied")
PYEOF
# (13) Aux probe CE in the actor update (consumes the 10I tensors). AUX_CE_BALANCE_MODE=off keeps
#      the aux-first per-micro-batch backward. norm_ratio does one optimizer update as:
#        1. accumulate unweighted aux gradients, measure their global norm, discard;
#        2. accumulate GRPO gradients and measure their global norm;
#        3. compute/synchronize c=min(cap,target*||gP||/(||gA||+eps));
#        4. replay the aux forwards under their saved RNG states with coefficient c;
#        5. measure the combined norm, then run verl's usual clip + optimizer step.
#      The three norms give the policy/aux cosine without keeping a gradient copy.
#      10I schedules aux rows positionally, so every rank runs identical forward/backward and norm
#      collectives; coefficient MIN/MAX consensus is checked before replay and the midpoint is used
#      on every rank. Per-term metrics are additive sums on every aux micro-batch, to match verl's
#      rank concatenation and mean reducer.
# dp_actor.py is reset to pristine first so an older (13) marker cannot short-circuit a newer patch.
if git -C "$VERL_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$VERL_DIR" checkout -- verl/workers/actor/dp_actor.py 2>/dev/null \
    && echo "[patch_verl] (13) reset dp_actor.py to pristine before patching" || true
fi
python - "$VERL_DIR" <<'PYEOF'
import sys
p = sys.argv[1] + "/verl/workers/actor/dp_actor.py"
s = open(p).read()
if "[persuasion-gym] (13T)" in s:
    print("[patch_verl] (13) aux-CE actor loss + norm balancing: already present")
    raise SystemExit(0)
if "[persuasion-gym] (13)" in s:
    raise SystemExit(
        "[patch_verl] (13) an older aux-CE block remains after reset. Restore "
        "verl/workers/actor/dp_actor.py from upstream, then re-run.")

# Read metadata before data.select(), which silently drops non-whitelisted keys.
anchor_a = (
    "        # Include pre-computed IS weights if present in batch\n"
    "        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True\n"
    '        if "rollout_is_weights" in data.batch.keys():\n'
    '            select_keys.append("rollout_is_weights")\n'
)
block_a = (
    "        # [persuasion-gym] (13T) aux-CE tensors + gradient-balance configuration.\n"
    "        _aux_keys = [\"aux_input_ids\", \"aux_attention_mask\", \"aux_position_ids\",\n"
    "                     \"aux_responses\", \"aux_response_mask\", \"aux_valid\", \"aux_term_id\"]\n"
    "        _aux_on = all(k in data.batch.keys() for k in _aux_keys)\n"
    "        _aux_coeff = float(data.meta_info.get(\"aux_ce_coeff\", 0.0)) if _aux_on else 0.0\n"
    "        _aux_terms = [t for t in str(data.meta_info.get(\"aux_ce_terms\", \"\")).split(\",\") if t]\n"
    "        _aux_balance_mode = str(data.meta_info.get(\"aux_ce_balance_mode\", \"off\")).strip()\n"
    "        _aux_balance_cfg = None\n"
    "        if _aux_on:\n"
    "            assert not self.config.use_dynamic_bsz, \"[aux-ce] use_dynamic_bsz unsupported\"\n"
    "            assert _aux_coeff > 0.0, \"[aux-ce] aux tensors present but aux_ce_coeff missing\"\n"
    "            assert _aux_terms, \"[aux-ce] aux tensors present but aux_ce_terms missing (stale 10I?)\"\n"
    "            assert data.batch[\"aux_responses\"].shape == data.batch[\"aux_response_mask\"].shape\n"
    "            assert int(data.batch[\"aux_term_id\"].max()) < len(_aux_terms), \\\n"
    "                \"[aux-ce] aux_term_id out of range for aux_ce_terms\"\n"
    "            assert _aux_balance_mode in (\"off\", \"norm_ratio\"), \\\n"
    "                f\"[aux-ce] unknown balance mode: {_aux_balance_mode}\"\n"
    "            if _aux_balance_mode == \"norm_ratio\":\n"
    "                import rl.aux_grad_balance as _aux_gb\n"
    "                _aux_balance_cfg = _aux_gb.config_from_mapping(\n"
    "                    {\"AUX_CE_BALANCE_MODE\": _aux_balance_mode,\n"
    "                     \"AUX_CE_TARGET_GRAD_RATIO\": str(\n"
    "                         data.meta_info.get(\"aux_ce_target_grad_ratio\", 0.25))},\n"
    "                    coeff_cap=_aux_coeff)\n"
    "                _aux_eps = float(data.meta_info.get(\n"
    "                    \"aux_ce_balance_epsilon\", _aux_balance_cfg.epsilon))\n"
    "                assert _aux_eps == _aux_balance_cfg.epsilon, \\\n"
    "                    \"[aux-ce] driver/actor balance epsilon mismatch\"\n"
    "            select_keys.extend(_aux_keys)\n"
)
assert s.count(anchor_a) == 1, \
    "[patch_verl] (13A) rollout_is_weights anchor not unique/found (verl version drift?)"
s = s.replace(anchor_a, anchor_a + block_a, 1)

# Nested helpers so the legacy and adaptive paths share one aux CE implementation.
anchor_u = (
    "        metrics = {}\n"
    "        for _ in range(self.config.ppo_epochs):\n"
)
block_u = (
    "        # How many optimizer steps may drop the aux term for a non-finite raw gradient\n"
    "        # before the run is failed. Small on purpose: a handful of skips out of 100 is a\n"
    "        # recoverable numerical hiccup, a large count is a structural problem that must\n"
    "        # not be averaged away silently. Every step emits actor/aux_skipped either way.\n"
    "        _aux_nonfinite_budget = int(os.environ.get(\"AUX_CE_NONFINITE_BUDGET\", \"5\"))\n"
    "\n"
    "        def _aux_grad_norm_no_clip(_name, _soft=False):\n"
    "            if isinstance(self.actor_module, FSDP):\n"
    "                _norm = self.actor_module.clip_grad_norm_(max_norm=float(\"inf\"))\n"
    "            elif isinstance(self.actor_module, FSDPModule):\n"
    "                _norm = fsdp2_clip_grad_norm_(\n"
    "                    self.actor_module.parameters(), max_norm=float(\"inf\"))\n"
    "            else:\n"
    "                _norm = torch.nn.utils.clip_grad_norm_(\n"
    "                    self.actor_module.parameters(), max_norm=float(\"inf\"))\n"
    "            if isinstance(_norm, DTensor):\n"
    "                _norm = _norm.full_tensor()\n"
    "            if not torch.is_tensor(_norm):\n"
    "                _norm = torch.tensor(float(_norm), device=get_device_id())\n"
    "            if not bool(torch.isfinite(_norm)) or float(_norm.detach().item()) < 0.0:\n"
    "                if _soft:\n"
    "                    return None\n"
    "                raise FloatingPointError(f\"[aux-ce] non-finite {_name}: {_norm}\")\n"
    "            return _norm\n"
    "\n"
    "        def _aux_sync_coefficient(_value):\n"
    "            _coef = torch.tensor(float(_value), dtype=torch.float64, device=get_device_id())\n"
    "            if torch.distributed.is_available() and torch.distributed.is_initialized():\n"
    "                _lo, _hi = _coef.clone(), _coef.clone()\n"
    "                torch.distributed.all_reduce(_lo, op=torch.distributed.ReduceOp.MIN)\n"
    "                torch.distributed.all_reduce(_hi, op=torch.distributed.ReduceOp.MAX)\n"
    "                _tol = 1e-12 + 1e-7 * max(abs(_lo.item()), abs(_hi.item()))\n"
    "                if (_hi - _lo).abs().item() > _tol:\n"
    "                    raise RuntimeError(\n"
    "                        f\"[aux-ce] rank-disagreeing coefficients: min={_lo.item()} max={_hi.item()}\")\n"
    "                _coef = (_lo + _hi) * 0.5\n"
    "            return float(_coef.item())\n"
    "\n"
    "        def _aux_capture_rng():\n"
    "            _cuda_state = (torch.cuda.get_rng_state(get_device_id())\n"
    "                           if torch.cuda.is_available() else None)\n"
    "            return torch.get_rng_state(), _cuda_state\n"
    "\n"
    "        def _aux_restore_rng(_state):\n"
    "            torch.set_rng_state(_state[0])\n"
    "            if _state[1] is not None:\n"
    "                torch.cuda.set_rng_state(_state[1], get_device_id())\n"
    "\n"
    "        def _aux_backward(_model_inputs, _coefficient, *, _strict, _emit_metrics):\n"
    "            _out = {}\n"
    "            _has_aux = bool(_model_inputs[\"aux_valid\"].any())\n"
    "            if (_strict and torch.distributed.is_available()\n"
    "                    and torch.distributed.is_initialized()):\n"
    "                _valid_count = torch.tensor(int(_has_aux), device=get_device_id())\n"
    "                torch.distributed.all_reduce(_valid_count, op=torch.distributed.ReduceOp.SUM)\n"
    "                _world = torch.distributed.get_world_size()\n"
    "                if _valid_count.item() not in (0, _world):\n"
    "                    raise RuntimeError(\n"
    "                        f\"[aux-ce] rank-divergent aux_valid: {_valid_count.item()}/{_world}\")\n"
    "                _has_aux = _valid_count.item() == _world\n"
    "            if not _has_aux:\n"
    "                return _out\n"
    "            _, _aux_logp = self._forward_micro_batch(\n"
    "                {\"input_ids\": _model_inputs[\"aux_input_ids\"],\n"
    "                 \"attention_mask\": _model_inputs[\"aux_attention_mask\"],\n"
    "                 \"position_ids\": _model_inputs[\"aux_position_ids\"],\n"
    "                 \"responses\": _model_inputs[\"aux_responses\"]},\n"
    "                temperature=1.0, calculate_entropy=False)\n"
    "            _aux_mask = _model_inputs[\"aux_response_mask\"].to(_aux_logp.dtype)\n"
    "            _aux_tok = _aux_mask.sum()\n"
    "            _aux_ce = -(_aux_logp * _aux_mask).sum() / _aux_tok.clamp(min=1.0)\n"
    "            if _strict and not bool(torch.isfinite(_aux_ce.detach())):\n"
    "                raise FloatingPointError(f\"[aux-ce] non-finite auxiliary loss: {_aux_ce}\")\n"
    "            (_coefficient * _aux_ce * (1.0 / self.gradient_accumulation)).backward()\n"
    "            if not _emit_metrics:\n"
    "                return _out\n"
    "            if _aux_tok.item() > 0:\n"
    "                _out[\"actor/aux_ce_loss\"] = _aux_ce.detach().item()\n"
    "                _aux_gp = ((_aux_logp.detach() * _aux_mask).sum()\n"
    "                           / _aux_tok.clamp(min=1.0)).exp().item()\n"
    "                _out[\"actor/aux_gold_prob\"] = _aux_gp\n"
    "                _out[\"actor/aux_acc\"] = float(_aux_gp > 0.5)\n"
    "            _aux_oh = (_model_inputs[\"aux_term_id\"].long().unsqueeze(-1)\n"
    "                       == torch.arange(len(_aux_terms), device=_aux_logp.device)\n"
    "                       ).to(_aux_logp.dtype)\n"
    "            _aux_rtok = _aux_mask.sum(-1)\n"
    "            _aux_rce = -(_aux_logp.detach() * _aux_mask).sum(-1)\n"
    "            _aux_rgp = (-_aux_rce / _aux_rtok.clamp(min=1.0)).exp()\n"
    "            _aux_pt = (torch.stack([\n"
    "                _aux_rce, _aux_rtok, (_aux_rtok > 0).to(_aux_logp.dtype),\n"
    "                ((_aux_rgp > 0.5) & (_aux_rtok > 0)).to(_aux_logp.dtype)]) @ _aux_oh).tolist()\n"
    "            for _ti, _tn in enumerate(_aux_terms):\n"
    "                _out[f\"actor/aux_ce_sum__{_tn}\"] = _aux_pt[0][_ti]\n"
    "                _out[f\"actor/aux_tok__{_tn}\"] = _aux_pt[1][_ti]\n"
    "                _out[f\"actor/aux_rows__{_tn}\"] = _aux_pt[2][_ti]\n"
    "                _out[f\"actor/aux_hit__{_tn}\"] = _aux_pt[3][_ti]\n"
    "            return _out\n"
    "\n"
)
assert s.count(anchor_u) == 1, \
    "[patch_verl] (13U) metrics anchor not unique/found (verl version drift?)"
s = s.replace(anchor_u, "        metrics = {}\n" + block_u
              + "        for _ in range(self.config.ppo_epochs):\n", 1)

# Adaptive pass 1: measure and discard the accumulated raw auxiliary gradient.
anchor_m = (
    "                self.actor_optimizer.zero_grad()\n"
    "\n"
    "                for micro_batch in micro_batches:\n"
)
block_m = (
    "                _aux_rng_states = []\n"
    "                _aux_raw_grad_norm = None\n"
    "                if _aux_on and _aux_balance_mode == \"norm_ratio\":\n"
    "                    # Make the diagnostic aux pass observationally RNG-neutral for the\n"
    "                    # policy objective. Saved per-micro-batch states still reproduce the\n"
    "                    # exact raw aux gradient during the weighted replay below.\n"
    "                    _pre_aux_rng = _aux_capture_rng()\n"
    "                    for _aux_micro_batch in micro_batches:\n"
    "                        _aux_micro_batch = _aux_micro_batch.to(get_device_id())\n"
    "                        _aux_inputs = {**_aux_micro_batch.batch,\n"
    "                                       **_aux_micro_batch.non_tensor_batch}\n"
    "                        _aux_rng_states.append(_aux_capture_rng())\n"
    "                        _aux_metrics = _aux_backward(\n"
    "                            _aux_inputs, 1.0, _strict=True, _emit_metrics=True)\n"
    "                        append_to_dict(metrics, _aux_metrics)\n"
    "                    # RECOVERABLE. A non-finite RAW aux gradient skips the aux term for this\n"
    "                    # optimizer step instead of killing the run: the policy update proceeds\n"
    "                    # unchanged and only the auxiliary contribution is dropped. This is the\n"
    "                    # one collective-safe recovery point -- clip_grad_norm_ all-reduces, so\n"
    "                    # every rank reads the SAME norm and takes the SAME branch. Catching a\n"
    "                    # per-rank failure inside the micro-batch loop above would instead let\n"
    "                    # ranks diverge mid-FSDP-collective and hang, which is why the strict\n"
    "                    # finite-LOSS check in _aux_backward stays fatal.\n"
    "                    _aux_raw_grad_norm = _aux_grad_norm_no_clip(\"aux_grad_norm\", _soft=True)\n"
    "                    if _aux_raw_grad_norm is None:\n"
    "                        self._aux_nonfinite_steps = (\n"
    "                            getattr(self, \"_aux_nonfinite_steps\", 0) + 1)\n"
    "                        if self._aux_nonfinite_steps > _aux_nonfinite_budget:\n"
    "                            raise FloatingPointError(\n"
    "                                \"[aux-ce] non-finite aux gradient on \"\n"
    "                                f\"{self._aux_nonfinite_steps} optimizer steps, over \"\n"
    "                                f\"AUX_CE_NONFINITE_BUDGET={_aux_nonfinite_budget}\")\n"
    "                    self.actor_optimizer.zero_grad()\n"
    "                    _aux_restore_rng(_pre_aux_rng)\n"
    "\n"
)
assert s.count(anchor_m) == 1, \
    "[patch_verl] (13M) optimizer-zero anchor not unique/found (verl version drift?)"
s = s.replace(anchor_m, "                self.actor_optimizer.zero_grad()\n\n"
              + block_m + "                for micro_batch in micro_batches:\n", 1)

# Legacy path: aux backward right before each policy forward.
anchor_b = (
    "                    micro_batch_metrics = {}\n"
    "                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}\n"
)
block_b = (
    "                    if _aux_on and _aux_balance_mode == \"off\":\n"
    "                        micro_batch_metrics.update(_aux_backward(\n"
    "                            model_inputs, _aux_coeff, _strict=False, _emit_metrics=True))\n"
)
assert s.count(anchor_b) == 1, \
    "[patch_verl] (13B) micro-batch anchor not unique/found (verl version drift?)"
s = s.replace(anchor_b, anchor_b + block_b, 1)

# Adaptive mode fails before backward on a non-finite policy objective.
anchor_f = (
    "                    else:\n"
    "                        loss = policy_loss * loss_scale_factor\n"
    "                    loss.backward()\n"
)
block_f = (
    "                    if (_aux_on and _aux_balance_mode == \"norm_ratio\"\n"
    "                            and not bool(torch.isfinite(loss.detach()))):\n"
    "                        raise FloatingPointError(f\"[aux-ce] non-finite policy loss: {loss}\")\n"
)
assert s.count(anchor_f) == 1, \
    "[patch_verl] (13F) policy backward anchor not unique/found (verl version drift?)"
s = s.replace(anchor_f, anchor_f.replace("                    loss.backward()\n", "")
              + block_f + "                    loss.backward()\n", 1)

# Adaptive pass 2: measure policy, synchronize coefficient, RNG-exact aux replay, diagnostics.
anchor_r = "                grad_norm = self._optimizer_step()\n"
block_r = (
    "                if _aux_on and _aux_balance_mode == \"norm_ratio\":\n"
    "                    _policy_grad_norm = _aux_grad_norm_no_clip(\"policy_grad_norm\")\n"
    "                    if _aux_raw_grad_norm is None:\n"
    "                        # Aux skipped this step: coefficient 0 suppresses the weighted replay\n"
    "                        # below, leaving a pure-policy update. Still routed through\n"
    "                        # _aux_sync_coefficient so the cross-rank agreement check is retained.\n"
    "                        _effective_coeff = 0.0\n"
    "                    else:\n"
    "                        _effective_coeff = _aux_gb.effective_aux_coefficient(\n"
    "                            _policy_grad_norm.detach().item(),\n"
    "                            _aux_raw_grad_norm.detach().item(), _aux_balance_cfg)\n"
    "                    _effective_coeff = _aux_sync_coefficient(_effective_coeff)\n"
    "                    if _effective_coeff > 0.0:\n"
    "                        _post_policy_rng = _aux_capture_rng()\n"
    "                        try:\n"
    "                            for _aux_micro_batch, _aux_rng in zip(\n"
    "                                    micro_batches, _aux_rng_states, strict=True):\n"
    "                                _aux_restore_rng(_aux_rng)\n"
    "                                _aux_micro_batch = _aux_micro_batch.to(get_device_id())\n"
    "                                _aux_inputs = {**_aux_micro_batch.batch,\n"
    "                                               **_aux_micro_batch.non_tensor_batch}\n"
    "                                _aux_backward(_aux_inputs, _effective_coeff,\n"
    "                                              _strict=True, _emit_metrics=False)\n"
    "                        finally:\n"
    "                            _aux_restore_rng(_post_policy_rng)\n"
    "                    _combined_grad_norm = _aux_grad_norm_no_clip(\"combined_grad_norm\")\n"
    "                    if _aux_raw_grad_norm is not None:\n"
    "                        _balance_metrics = _aux_gb.gradient_diagnostics(\n"
    "                            policy_norm=_policy_grad_norm.detach().item(),\n"
    "                            aux_norm=_aux_raw_grad_norm.detach().item(),\n"
    "                            coefficient=_effective_coeff,\n"
    "                            combined_norm=_combined_grad_norm.detach().item(),\n"
    "                            grad_clip=float(self.config.grad_clip),\n"
    "                            epsilon=_aux_balance_cfg.epsilon)\n"
    "                        append_to_dict(metrics, {\n"
    "                            f\"actor/{_key}\": _value\n"
    "                            for _key, _value in _balance_metrics.items()})\n"
    "                    # Emitted on EVERY adaptive step, skipped or not, so a skip is visible as\n"
    "                    # a value rather than as a hole in the series. All ranks agree on the\n"
    "                    # branch, so the per-rank metric key sets stay identical.\n"
    "                    append_to_dict(metrics, {\n"
    "                        \"actor/aux_skipped\": float(_aux_raw_grad_norm is None),\n"
    "                        \"actor/aux_nonfinite_steps\": float(\n"
    "                            getattr(self, \"_aux_nonfinite_steps\", 0))})\n"
    "\n"
)
assert s.count(anchor_r) == 1, \
    "[patch_verl] (13R) optimizer-step anchor not unique/found (verl version drift?)"
s = s.replace(anchor_r, block_r + anchor_r, 1)

open(p, "w").write(s)
print("[patch_verl] (13T) aux-CE actor loss: applied "
      "(legacy off + adaptive global-norm measure/replay)")
PYEOF
# (14) Deterministic per-request/per-turn SGLang sampling for paired rollout comparisons (opt-in
#      ROLLOUT_SEED; unset leaves verl unchanged). The trainer seeds each repeated training row from
#      (ROLLOUT_SEED, global step, extra_info[index], rollout offset); the schema carries it into
#      SGLang, which derives a sampling_seed per assistant turn. The engine fallback RNG and
#      deterministic-inference kernels are pinned too. 14R/14T are reapplied on every run (their
#      files are reset above); 14S is marker-guarded since sglang_rollout.py is not reset.
python - "$VERL_DIR" <<'PYEOF'
import sys
from pathlib import Path

root = Path(sys.argv[1])

# (14R) Request schema: retain the stable request seed across all turns/deep copies.
p = root / "verl/workers/rollout/schemas.py"
s = p.read_text()
if "[persuasion-gym] (14R)" in s:
    print("[patch_verl] (14R) rollout request seed schema: already present")
else:
    anchor = "    rollout_offset: int = 0\n    request_id: str\n"
    assert s.count(anchor) == 1, \
        "[patch_verl] (14R) AsyncRolloutRequest anchor not unique/found (verl version drift?)"
    repl = (
        "    rollout_offset: int = 0\n"
        "    # [persuasion-gym] (14R) stable training-request seed, reused via per-turn derivation.\n"
        "    rollout_sampling_seed: Optional[int] = None\n"
        "    request_id: str\n"
    )
    p.write_text(s.replace(anchor, repl, 1))
    print("[patch_verl] (14R) rollout request seed schema: applied")

# (14T) Stamp repeated training rows in exactly DataProto.repeat(interleave=True) order.
p = root / "verl/trainer/ppo/ray_trainer.py"
s = p.read_text()
if "[persuasion-gym] (14T)" in s:
    print("[patch_verl] (14T) deterministic training rollout seeds: already present")
else:
    anchor = (
        "                gen_batch_output = gen_batch.repeat(\n"
        "                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True\n"
        "                )\n"
    )
    assert s.count(anchor) == 1, \
        "[patch_verl] (14T) repeated-generation anchor not unique/found (verl version drift?)"
    block = (
        "\n"
        "                # [persuasion-gym] (14T) Common-random-number seeds for paired TRAINING\n"
        "                # rollouts. DataProto.repeat(interleave=True) orders rows as game-major then\n"
        "                # rollout offset, exactly matching interleaved_rollout_seeds(). Validation is\n"
        "                # deliberately unchanged; it never enters an optimizer update.\n"
        "                _rollout_seed_raw = os.environ.get(\"ROLLOUT_SEED\", \"\").strip()\n"
        "                if _rollout_seed_raw:\n"
        "                    _rollout_extra = batch.non_tensor_batch.get(\"extra_info\")\n"
        "                    if _rollout_extra is None or len(_rollout_extra) != len(batch):\n"
        "                        raise RuntimeError(\n"
        "                            \"ROLLOUT_SEED is enabled but training extra_info rows are missing\"\n"
        "                        )\n"
        "                    try:\n"
        "                        _rollout_game_ids = [_row[\"index\"] for _row in _rollout_extra]\n"
        "                    except (KeyError, TypeError) as _exc:\n"
        "                        raise RuntimeError(\n"
        "                            \"ROLLOUT_SEED is enabled but a training extra_info row lacks index\"\n"
        "                        ) from _exc\n"
        "                    import rl.rollout_seed as _rollout_seed_14t\n"
        "\n"
        "                    _rollout_n = int(self.config.actor_rollout_ref.rollout.n)\n"
        "                    _rollout_seeds = _rollout_seed_14t.interleaved_rollout_seeds(\n"
        "                        _rollout_game_ids, repeat_times=_rollout_n,\n"
        "                        base_seed=_rollout_seed_raw, global_step=self.global_steps\n"
        "                    )\n"
        "                    if len(_rollout_seeds) != len(gen_batch_output):\n"
        "                        raise RuntimeError(\n"
        "                            f\"rollout seed count {len(_rollout_seeds)} does not match \"\n"
        "                            f\"repeated training rows {len(gen_batch_output)}\"\n"
        "                        )\n"
        "                    gen_batch_output.non_tensor_batch[\"rollout_sampling_seed\"] = np.asarray(\n"
        "                        _rollout_seeds, dtype=np.int64\n"
        "                    )\n"
        "                    print(\n"
        "                        f\"[rollout-seed] step={self.global_steps} rows={len(_rollout_seeds)} \"\n"
        "                        f\"digest={_rollout_seed_14t.seed_digest(_rollout_seeds)}\", flush=True\n"
        "                    )\n"
        "                    _rollout_seed_dir = os.environ.get(\n"
        "                        \"ROLLOUT_SEED_PROVENANCE_DIR\", \"\"\n"
        "                    ).strip()\n"
        "                    if _rollout_seed_dir:\n"
        "                        os.makedirs(_rollout_seed_dir, exist_ok=True)\n"
        "                        _rollout_seed_path = os.path.join(\n"
        "                            _rollout_seed_dir, f\"step_{self.global_steps:05d}.json\"\n"
        "                        )\n"
        "                        _rollout_seed_tmp = _rollout_seed_path + f\".tmp.{os.getpid()}\"\n"
        "                        with open(_rollout_seed_tmp, \"w\") as _handle:\n"
        "                            json.dump({\n"
        "                                \"schema_version\": 1,\n"
        "                                \"global_step\": int(self.global_steps),\n"
        "                                \"game_ids\": [str(_id) for _id in _rollout_game_ids],\n"
        "                                \"repeat_times\": _rollout_n,\n"
        "                                \"seeds\": [int(_seed) for _seed in _rollout_seeds],\n"
        "                                \"digest\": _rollout_seed_14t.seed_digest(_rollout_seeds),\n"
        "                            }, _handle, sort_keys=True)\n"
        "                        os.replace(_rollout_seed_tmp, _rollout_seed_path)\n"
    )
    p.write_text(s.replace(anchor, anchor + block, 1))
    print("[patch_verl] (14T) deterministic training rollout seeds: applied")


# (14TF) Stamp the generation request seed on copies of the repeated extra_info rows, so the rare
# reward-side Receiver re-query is deterministic without mutating rows aliased by DataProto.repeat.
p = root / "verl/trainer/ppo/ray_trainer.py"
s = p.read_text()
if "[persuasion-gym] (14TF)" in s:
    print("[patch_verl] (14TF) deterministic reward-fallback seeds: already present")
else:
    fallback_anchor = (
        "                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)\n"
        "                    batch = batch.union(gen_batch_output)\n"
    )
    assert s.count(fallback_anchor) == 1, \
        "[patch_verl] (14TF) repeated-batch union anchor not unique/found (verl version drift?)"
    fallback_repl = (
        "                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)\n"
        "                    # [persuasion-gym] (14TF) seed exceptional reward-side re-queries.\n"
        "                    if _rollout_seed_raw:\n"
        "                        _rollout_fallback_rows = batch.non_tensor_batch.get(\"extra_info\")\n"
        "                        if _rollout_fallback_rows is None or len(_rollout_fallback_rows) != len(_rollout_seeds):\n"
        "                            raise RuntimeError(\"repeated extra_info rows do not match rollout seeds\")\n"
        "                        _rollout_fallback_rows = [\n"
        "                            {**dict(_row), \"rollout_sampling_seed\": int(_seed)}\n"
        "                            for _row, _seed in zip(\n"
        "                                _rollout_fallback_rows, _rollout_seeds, strict=True\n"
        "                            )\n"
        "                        ]\n"
        "                        batch.non_tensor_batch[\"extra_info\"] = np.asarray(\n"
        "                            _rollout_fallback_rows, dtype=object\n"
        "                        )\n"
        "                    batch = batch.union(gen_batch_output)\n"
    )
    p.write_text(s.replace(fallback_anchor, fallback_repl, 1))
    print("[patch_verl] (14TF) deterministic reward-fallback seeds: applied")
# (14S) Pin engine initialization, validate/carry the request seed, and set a fresh deterministic
# sampling_seed immediately before every assistant-turn engine call.
p = root / "verl/workers/rollout/sglang_rollout/sglang_rollout.py"
s = p.read_text()
if "[persuasion-gym] (14S)" in s:
    deterministic = 'args["enable_deterministic_inference"] = True'
    if deterministic in s:
        print("[patch_verl] (14S) deterministic SGLang rollout seeds: already present")
    else:
        old_engine_seed = (
            "            # [persuasion-gym] (14S) Pin SGLang's fallback RNG per stable worker rank.\n"
            "            _rollout_seed_raw_14s = os.environ.get(\"ROLLOUT_SEED\", \"\").strip()\n"
            "            if _rollout_seed_raw_14s:\n"
            "                import rl.rollout_seed as _rollout_seed_14s\n"
            "\n"
            "                args[\"random_seed\"] = _rollout_seed_14s.derive_engine_seed(\n"
            "                    base_seed=_rollout_seed_raw_14s, worker_rank=dist.get_rank()\n"
            "                )\n"
        )
        new_engine_seed = (
            "            # [persuasion-gym] (14S) Pin every engine to the same fallback RNG and use\n"
            "            # schedule-invariant kernels; request-level seeds separate all samples.\n"
            "            _rollout_seed_raw_14s = os.environ.get(\"ROLLOUT_SEED\", \"\").strip()\n"
            "            if _rollout_seed_raw_14s:\n"
            "                import rl.rollout_seed as _rollout_seed_14s\n"
            "\n"
            "                args[\"random_seed\"] = _rollout_seed_14s.derive_engine_seed(\n"
            "                    base_seed=_rollout_seed_raw_14s, worker_rank=0\n"
            "                )\n"
            "                args[\"enable_deterministic_inference\"] = True\n"
        )
        assert s.count(old_engine_seed) == 1, (
            "[patch_verl] (14S-upgrade) legacy engine-seed block not unique/found"
        )
        p.write_text(s.replace(old_engine_seed, new_engine_seed, 1))
        print("[patch_verl] (14S) deterministic SGLang rollout seeds: upgraded")
else:
    engine_anchor = "            }\n\n            if is_server_mode:\n"
    assert s.count(engine_anchor) == 1, \
        "[patch_verl] (14S-engine) engine-args anchor not unique/found (verl version drift?)"
    engine_repl = (
        "            }\n"
        "            # [persuasion-gym] (14S) Pin every engine to the same fallback RNG and use\n"
        "            # schedule-invariant kernels; request-level seeds separate all samples.\n"
        "            _rollout_seed_raw_14s = os.environ.get(\"ROLLOUT_SEED\", \"\").strip()\n"
        "            if _rollout_seed_raw_14s:\n"
        "                import rl.rollout_seed as _rollout_seed_14s\n"
        "\n"
        "                args[\"random_seed\"] = _rollout_seed_14s.derive_engine_seed(\n"
        "                    base_seed=_rollout_seed_raw_14s, worker_rank=0\n"
        "                )\n"
        "                args[\"enable_deterministic_inference\"] = True\n"
        "\n"
        "            if is_server_mode:\n"
    )
    s = s.replace(engine_anchor, engine_repl, 1)

    preprocess_anchor = (
        "        req_list = []\n"
        "        multi_modal_data_list = prompts.non_tensor_batch.get(\n"
        "            \"multi_modal_data\", [None] * len(prompts.non_tensor_batch[\"raw_prompt\"])\n"
        "        )\n"
    )
    assert s.count(preprocess_anchor) == 1, \
        "[patch_verl] (14S-preprocess) request-list anchor not unique/found (verl version drift?)"
    preprocess_block = (
        "        _rollout_seed_rows = prompts.non_tensor_batch.get(\"rollout_sampling_seed\")\n"
        "        _rollout_seed_enabled = bool(os.environ.get(\"ROLLOUT_SEED\", \"\").strip())\n"
        "        _rollout_seed_validate = bool(prompts.meta_info.get(\"validate\", False))\n"
        "        if _rollout_seed_enabled and not _rollout_seed_validate and _rollout_seed_rows is None:\n"
        "            raise RuntimeError(\n"
        "                \"ROLLOUT_SEED is enabled but SGLang received unseeded training rows\"\n"
        "            )\n"
        "        if _rollout_seed_rows is not None and len(_rollout_seed_rows) != len(\n"
        "            prompts.non_tensor_batch[\"raw_prompt\"]\n"
        "        ):\n"
        "            raise RuntimeError(\"rollout_sampling_seed length does not match SGLang requests\")\n"
    )
    s = s.replace(preprocess_anchor, preprocess_anchor + preprocess_block, 1)

    request_anchor = (
        "                batch_data_id=data_idx,\n"
        "                rollout_offset=0,\n"
        "                request_id=str(uuid4()),\n"
    )
    assert s.count(request_anchor) == 1, \
        "[patch_verl] (14S-request) request-constructor anchor not unique/found (verl version drift?)"
    request_repl = (
        "                batch_data_id=data_idx,\n"
        "                rollout_offset=0,\n"
        "                rollout_sampling_seed=(\n"
        "                    int(_rollout_seed_rows[data_idx]) if _rollout_seed_rows is not None else None\n"
        "                ),\n"
        "                request_id=str(uuid4()),\n"
    )
    s = s.replace(request_anchor, request_repl, 1)

    turn_anchor = (
        "                output = await self._handle_engine_call(\n"
        "                    _req, request_sampling_params, image_data=image_data\n"
        "                )\n"
    )
    # The pinned verl (6dc50993) has this call on one line; the three-line form is also accepted.
    turn_anchor_one_line = (
        "                output = await self._handle_engine_call(_req, request_sampling_params, image_data=image_data)\n"
    )
    if s.count(turn_anchor) == 1:
        chosen_turn_anchor = turn_anchor
    elif s.count(turn_anchor_one_line) == 1:
        chosen_turn_anchor = turn_anchor_one_line
    else:
        raise AssertionError(
            "[patch_verl] (14S-turn) engine-call anchor not unique/found (verl version drift?)"
        )
    turn_block = (
        "                if _req.rollout_sampling_seed is not None:\n"
        "                    import rl.rollout_seed as _rollout_seed_14s\n"
        "\n"
        "                    request_sampling_params[\"sampling_seed\"] = _rollout_seed_14s.derive_turn_seed(\n"
        "                        rollout_seed=_req.rollout_sampling_seed, turn_index=current_turns\n"
        "                    )\n"
        "                elif os.environ.get(\"ROLLOUT_SEED\", \"\").strip() and not is_validate:\n"
        "                    raise RuntimeError(\n"
        "                        \"ROLLOUT_SEED is enabled but a training request lacks rollout_sampling_seed\"\n"
        "                    )\n"
    )
    s = s.replace(chosen_turn_anchor, turn_block + chosen_turn_anchor, 1)
    p.write_text(s)
    print("[patch_verl] (14S) deterministic SGLang rollout seeds: applied")

# (14SR) Carry the request seed into the interaction on a per-request dict copy. DataProto.repeat
# aliases object-valued rows, so mutating the original dict would overwrite sibling rollouts.
p = root / "verl/workers/rollout/sglang_rollout/sglang_rollout.py"
s = p.read_text()
if "[persuasion-gym] (14SR)" in s:
    print("[patch_verl] (14SR) deterministic local-receiver seeds: already present")
else:
    interaction_anchor = (
        "            if self.interaction_map:\n"
        "                _interaction_kwargs = prompts.non_tensor_batch[\"interaction_kwargs\"][data_idx]\n"
        "            else:\n"
        "                _interaction_kwargs = {}\n"
    )
    assert s.count(interaction_anchor) == 1, \
        "[patch_verl] (14SR) interaction-kwargs anchor not unique/found (verl version drift?)"
    interaction_repl = (
        "            if self.interaction_map:\n"
        "                # [persuasion-gym] (14SR) np.repeat aliases object rows: copy before stamping.\n"
        "                _interaction_kwargs = dict(\n"
        "                    prompts.non_tensor_batch[\"interaction_kwargs\"][data_idx]\n"
        "                )\n"
        "                if _rollout_seed_rows is not None:\n"
        "                    _interaction_kwargs[\"rollout_sampling_seed\"] = int(\n"
        "                        _rollout_seed_rows[data_idx]\n"
        "                    )\n"
        "            else:\n"
        "                _interaction_kwargs = {}\n"
    )
    p.write_text(s.replace(interaction_anchor, interaction_repl, 1))
    print("[patch_verl] (14SR) deterministic local-receiver seeds: applied")
PYEOF
echo "[patch_verl] done: $VERL_DIR"
