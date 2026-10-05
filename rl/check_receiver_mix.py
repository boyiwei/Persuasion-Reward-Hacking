"""Offline checks for the mixture-of-receivers routing and transport layer (rl/receiver_client.py).

Covers per-model request-body rules, arm discovery, resolve_backend precedence and its refusal to
degrade, RECEIVER_MIX_PIN, and the per-arm timeout/semaphore split. No network or GPU needed.
Allocation logic is covered by `python -m rl.receiver_mix --selftest`.

    python -m rl.check_receiver_mix          # offline checks only
    python -m rl.check_receiver_mix --live   # + one real completion per hosted arm

--live is the Stage-B gate: it POSTs the exact runtime body (api_body) for each hosted arm and
requires non-empty content. A preflight body missing a runtime parameter (e.g. top_p) can pass
while every runtime call gets HTTP 400, and a reasoning model at small max_tokens returns "choices"
with empty content (finish_reason="length"), which a bare `grep '"choices"'` would accept.
"""
import argparse
import collections
import os
import sys


def _key_from_bashrc() -> str:
    """Recover the bearer key from ~/.bashrc when the environment lacks it.

    Non-interactive shells (sbatch, `bash script.sh`) skip the interactive part of ~/.bashrc, so
    GATEWAY_API_KEY is often unset; scripts/rl_train_sender.slurm does the same grep. Also sets the
    variable so later rc._gateway_key() calls find it."""
    import re as _re
    name = os.getenv("RECEIVER_MIX_KEY_ENV", "GATEWAY_API_KEY") or "GATEWAY_API_KEY"
    try:
        with open(os.path.expanduser("~/.bashrc")) as fh:
            for line in fh:
                m = _re.match(rf"\s*export\s+{_re.escape(name)}=(.*)", line)
                if m:
                    val = m.group(1).strip().strip('"').strip("'")
                    if val:
                        os.environ[name] = val
                        return val
    except OSError:
        pass
    return ""


def _reset(rc, **env):
    """Hermetic env for the offline assertions.

    Also pops RECEIVER_MIX_SEED: the launcher runs these checks inside the job env, and a job seed
    would shift the hash-split counts asserted below."""
    for k in ("RECEIVER_MIX_SPEC", "RECEIVER_MIX_MODEL", "RECEIVER_MIX_PIN", "RECEIVER_MIX_FRAC",
              "RECEIVER_MIX_SEED"):
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in env.items() if v is not None})
    rc._labels_cached.cache_clear()


def offline() -> None:  # noqa: C901 - a flat list of independent assertions
    from rl import receiver_client as rc

    def ok(name):
        print(f"  ok  {name}")

    spec = "local:4,DeepSeek-V4-Flash:4,gpt-5-mini:4,grok-4-1-fast-reasoning:4"
    # Pass gateway_url explicitly and clear GATEWAY_URL and GATEWAY_TIER: at job start the ambient
    # RECEIVER_MIX_GATEWAY_URL may be a loopback sentinel, and the job's own GATEWAY_URL or tier
    # would change the tier assertions. The job's values are restored before --live runs.
    gw = "http://localhost:18742/gateway/chat/completions"   # a relay that forwards to the gateway
    saved = {k: os.environ.pop(k, None) for k in ("GATEWAY_URL", "GATEWAY_TIER")}
    try:
        # 1. api_body per-model rules.
        b = rc.api_body("gpt-5-mini", [], 8192, gateway_url=gw)
        assert "top_p" not in b, b
        assert b["temperature"] == 1.0, b
        assert b["tier"] == "base", b
        assert "reasoning_effort" not in b, "full reasoning is the chosen config; see _API_PARAM_RULES"
        for m in ("grok-4-1-fast-reasoning", "DeepSeek-V4-Flash", "qwen3.5-35B"):
            assert "top_p" in rc.api_body(m, [], 8192, gateway_url=gw), m
        assert "top_p" not in rc.api_body("o3-mini", [], 8192, gateway_url=gw)
        for other in ("https://openrouter.ai/api/v1/chat/completions",
                      "http://127.0.0.1:30001/v1/chat/completions"):
            assert "tier" not in rc.api_body("gpt-5-mini", [], 8192, gateway_url=other), other
        # GATEWAY_URL itself gets the tier whatever its spelling.
        os.environ["GATEWAY_URL"] = "https://api.example.com/v1/chat/completions"
        b3 = rc.api_body("gpt-5-mini", [], 8192, gateway_url=os.environ["GATEWAY_URL"])
        assert b3.get("tier") == "base", b3
        os.environ.pop("GATEWAY_URL")
        os.environ["RL_RECV_API_DROP_DEEPSEEK_V4_FLASH"] = "top_p,max_tokens"
        b2 = rc.api_body("DeepSeek-V4-Flash", [], 8192, gateway_url=gw)
        assert "top_p" not in b2 and "max_tokens" not in b2, b2
        del os.environ["RL_RECV_API_DROP_DEEPSEEK_V4_FLASH"]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    ok("api_body: gpt-5/o-series drop top_p + pin temperature=1.0; others keep it; "
       "tier sent only to GATEWAY_URL or a /gateway/ relay; per-arm env drop-hatch")

    # 2. Feature off.
    _reset(rc)
    assert rc.mix_labels() == [] and rc.resolve_backend(1) == "local"
    ok("feature off -> 'local', no arms announced")

    # 3. Legacy 2-way, unchanged.
    _reset(rc, RECEIVER_MIX_MODEL="DeepSeek-V4-Flash", RECEIVER_MIX_FRAC="0.5")
    assert rc.mix_labels() == ["api", "local"]
    assert rc.resolve_backend(1, "api") == "api" and rc.resolve_backend(1, "local") == "local"
    legacy = [rc.resolve_backend(g) for g in range(400)]
    assert set(legacy) == {"api", "local"}
    assert 150 < legacy.count("api") < 250, legacy.count("api")
    assert rc.nway_labels() == [], "legacy mode must emit NO per-arm reward-extras"
    ok(f"legacy 2-way: labels ['api','local'], nway_labels()=[] (frozen key set), "
       f"unstamped hash split {legacy.count('api')}/400 api")

    # 4. N-way arm discovery + stamped routing + unstamped hash buckets.
    _reset(rc, RECEIVER_MIX_SPEC=spec)
    assert rc.mix_labels() == ["local", "DeepSeek-V4-Flash", "gpt-5-mini",
                               "grok-4-1-fast-reasoning"], rc.mix_labels()
    for lab in rc.mix_labels():
        assert rc.resolve_backend(1, lab) == lab
    assert rc.nway_labels() == rc.mix_labels(), "N-way mode must emit per-arm reward-extras"
    c = collections.Counter(rc.resolve_backend(g) for g in range(4000))
    assert set(c) == set(rc.mix_labels()), c
    assert all(800 < v < 1200 for v in c.values()), c
    ok(f"N-way: 4 arms discovered; stamped routing exact; unstamped buckets {dict(c)}")

    # 5. Unknown or legacy stamps raise under an N-way spec rather than falling back to 'local'.
    for bad, why in [("api", "legacy sentinel under an N-way spec"),
                     ("gpt-5-nano", "unconfigured arm"),
                     ("", "empty string")]:
        try:
            rc.resolve_backend(1, bad)
        except ValueError:
            continue
        raise AssertionError(f"resolve_backend({bad!r}) must raise -- {why}")
    ok("unknown and legacy-'api' stamps RAISE (no silent degrade to local)")

    # 6. RECEIVER_MIX_PIN.
    os.environ["RECEIVER_MIX_PIN"] = "grok-4-1-fast-reasoning"
    assert all(rc.resolve_backend(g) == "grok-4-1-fast-reasoning" for g in range(50))
    assert rc.resolve_backend(1, "local") == "local", "an explicit stamp must still beat the pin"
    os.environ["RECEIVER_MIX_PIN"] = "nope"
    try:
        rc.resolve_backend(1)
    except ValueError:
        pass
    else:
        raise AssertionError("a pin outside the configured arm set must raise")
    os.environ.pop("RECEIVER_MIX_PIN")
    ok("RECEIVER_MIX_PIN pins unstamped rows, loses to an explicit stamp, and is validated")

    # 7. Per-arm timeout / semaphore split.
    os.environ["RL_RECV_API_TIMEOUT"] = "240"
    os.environ["RL_RECV_API_TIMEOUT_GROK_4_1_FAST_REASONING"] = "300"
    assert rc._env_for_label("RL_RECV_API_TIMEOUT", "grok-4-1-fast-reasoning", 120.0) == 300.0
    assert rc._env_for_label("RL_RECV_API_TIMEOUT", "gpt-5-mini", 120.0) == 240.0
    assert rc._env_for_label("RL_RECV_API_TIMEOUT", "gpt-5-mini", 120.0) == 240.0
    for k in ("RL_RECV_API_TIMEOUT", "RL_RECV_API_TIMEOUT_GROK_4_1_FAST_REASONING"):
        os.environ.pop(k)
    assert rc._api_semaphore("gpt-5-mini") is not rc._api_semaphore("grok-4-1-fast-reasoning")
    assert rc._api_semaphore("gpt-5-mini") is rc._api_semaphore("gpt-5-mini")
    ok("per-arm RL_RECV_API_TIMEOUT_<ARM> override; distinct per-arm semaphores")

    print("\nreceiver_client offline checks: ALL PASSED")


def live(spec: str, url: str, attempts: int = 4, max_tokens: int = 2048) -> int:
    """POST one real completion per hosted arm with the runtime body; require non-empty text.

    Retries `attempts` times per arm, 10 s apart: a loopback target that only passed the
    /v1/models poll still pays JIT / cuda-graph warm-up on its first completion (a 30 s single
    ping is too short), and the shared gateway throws transient 502s (a few percent of calls)."""
    import json
    import time

    import requests

    from rl import receiver_client as rc
    from rl import receiver_mix

    os.environ["RECEIVER_MIX_GATEWAY_URL"] = url
    key = rc._gateway_key() or _key_from_bashrc()
    if not key:
        print(f"FAIL: bearer key env {os.getenv('RECEIVER_MIX_KEY_ENV', 'GATEWAY_API_KEY')} is empty")
        return 1
    msgs = [{"role": "system", "content":
             "You are a skeptical juror. Respond EXACTLY as:\n<belief>0..1</belief>\n"
             "<argument>one short paragraph</argument>\nYour prior is 0.10."},
            {"role": "user", "content":
             "ROUND 3 of 3. Evidence: the stolen watch was found under the defendant's bed; a "
             "neighbour saw a similar man but could not see his face; the employer says the "
             "defendant worked until eight and the theft was at nine. State your updated belief."}]
    bad = 0
    for label in receiver_mix.hosted_labels(spec):
        body = rc.api_body(label, msgs, max_tokens, gateway_url=url)
        why = "no attempt made"
        for attempt in range(1, attempts + 1):
            t0 = time.time()
            try:
                r = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                                "Content-Type": "application/json"},
                                  json=body, timeout=300)
            except Exception as e:  # noqa: BLE001
                why = f"transport error: {e}"
                r = None
            if r is not None:
                dt = time.time() - t0
                if r.status_code != 200:
                    why = (f"HTTP {r.status_code}\n       body sent: {json.dumps(body)[:220]}"
                           f"\n       response:  {r.text[:400]}")
                else:
                    try:
                        j = r.json()
                        content = (j["choices"][0]["message"]["content"] or "").strip()
                    except Exception as e:  # noqa: BLE001
                        why = f"unparseable response ({e}): {r.text[:300]}"
                        content = None
                    if content:
                        has = "<belief>" in content.lower()
                        print(f"  {'ok  ' if has else 'WARN'} {label:26} {dt:5.1f}s 200 "
                              f"len={len(content)} belief_tag={has}"
                              f"{'' if attempt == 1 else f' (attempt {attempt})'}")
                        if not has:
                            bad += 1
                        why = None
                        break
                    if content == "":
                        # "choices" present but empty text: a reasoning model spent the whole
                        # budget on hidden reasoning tokens.
                        fin = j["choices"][0].get("finish_reason")
                        why = (f"200 but EMPTY content (finish_reason={fin}) -- a reasoning model "
                               f"consumed the whole {max_tokens}-token budget; raise max_tokens")
            if attempt < attempts:
                print(f"  ..   {label:26} attempt {attempt}/{attempts} failed ({str(why)[:120]}); "
                      "retrying in 10s")
                time.sleep(10)
        if why:
            print(f"  FAIL {label:26} {why}")
            bad += 1
    print("\nlive arm probe: " + ("ALL PASSED" if not bad else f"{bad} ARM(S) FAILED"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                    help="also POST one real completion per hosted arm (needs a reachable gateway)")
    ap.add_argument("--spec", default=os.getenv("RECEIVER_MIX_SPEC", "")
                    or "local:4,DeepSeek-V4-Flash:4,gpt-5-mini:4,grok-4-1-fast-reasoning:4")
    ap.add_argument("--url", default=os.getenv("RECEIVER_MIX_GATEWAY_URL", "")
                    or os.getenv("GATEWAY_URL", ""),
                    help="chat-completions URL for --live (default RECEIVER_MIX_GATEWAY_URL, else "
                         "GATEWAY_URL), e.g. a relay at "
                         "http://localhost:18742/gateway/chat/completions")
    ap.add_argument("--attempts", type=int, default=4, help="retries per arm in --live")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="--live completion budget; must exceed a reasoning model's hidden budget")
    ap.add_argument("--skip-offline", action="store_true")
    a = ap.parse_args()
    if a.live and not a.url:
        ap.error("--live needs a gateway URL: GATEWAY_URL is unset or empty (export the API "
                 "gateway's chat-completions URL, or pass --url)")
    if not a.skip_offline:
        offline()
    if a.live:
        print(f"\nlive arm probe via {a.url}")
        return live(a.spec, a.url, attempts=a.attempts, max_tokens=a.max_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
