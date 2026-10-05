"""OpenAI-compatible client for the served Qwen3.5-35B-A3B in both roles: receiver (the juror,
whose P(guilty) drives the reward) and judge (the reward-hacking monitors).

Mirrors agents/model/openai_llm.py::SGLangModel but addresses RECEIVER_HOST/RECEIVER_PORT (falling
back to SGLANG_HOST/SGLANG_PORT), so the judge server is independent of any policy-rollout sglang
engine verl runs. Env vars and keys are read at call time.

chat() serves both roles with the model-card sampling params (temperature=1.0, top_p=0.95,
top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0), each overridable by its
RL_RECV_* env var or per call. Native <think> stays off (enable_thinking=False) so the 35B answers
within max_tokens; thinking-block stripping lives in the interaction/reward/transcript code.

receiver_chat() is the receiver-role entry point and can route a game to a hosted API engine
(mixture of receivers, see below). chat() never routes, so the judge never mixes.
"""
import contextvars
import functools
import hashlib
import os
import re
import threading
import time

import requests
from openai import OpenAI

from rl import receiver_mix

_DEFAULT_MODEL = os.getenv("RECEIVER_MODEL_ID", "qwen3.5-35B")

# The game a chat() call belongs to, for offline drivers that record calls to a judge dump. The
# per-game workers of the offline audits (rl.strategy_audit.run, evaluation.audit_fabrications)
# set()/reset() it inside the worker function, since ThreadPoolExecutor does not propagate
# contexts. Records with no id (None) are dumped as game_id null / n_unattributed, never guessed
# from prompt text (identical round-1 arguments would merge games). Not read in this module; the
# online path never sets it.
current_game_id: contextvars.ContextVar = contextvars.ContextVar("receiver_client.current_game_id",
                                                                 default=None)


def local_endpoint() -> tuple:
    """(host, port) of the local served model: RECEIVER_HOST/RECEIVER_PORT, else
    SGLANG_HOST/SGLANG_PORT, else 127.0.0.1:30001. Drivers use it to name the transport in failure
    messages (evaluation/audit_fabrications.py)."""
    host = os.getenv("RECEIVER_HOST", os.getenv("SGLANG_HOST", "127.0.0.1"))
    port = os.getenv("RECEIVER_PORT", os.getenv("SGLANG_PORT", "30001"))
    return host, port

# --- Mixture-of-receivers: hosted API receivers via the API gateway ----------------------------
# Receiver role only; the judge stays on the local model (rl/monitors.py calls chat() directly).
#   2-way: RECEIVER_MIX_MODEL (e.g. "DeepSeek-V4-Flash") + RECEIVER_MIX_FRAC; labels "api"/"local",
#     run-name tag `_mix-<slug><frac>`.
#   N-way: RECEIVER_MIX_SPEC='local:4,DeepSeek-V4-Flash:4,gpt-5-mini:4,grok-4-1-fast-reasoning:4';
#     labels are hosted model ids, "local" is the co-served model. All hosted arms share one
#     gateway URL and key; the relay forwards the body verbatim, so arms differ only in "model".
# Both empty = off, and receiver_chat() == chat(). Per-game arm precedence: see resolve_backend().
_API_SEMS = {}
_API_SEM_LOCK = threading.Lock()
_WARNED = set()
_WARN_LOCK = threading.Lock()


def _warn_once(msg: str) -> None:
    """Print `msg` at most once per process (for per-call conditions such as a forced sampling
    param or a dropped kwarg, which would otherwise repeat ~38k times per job)."""
    with _WARN_LOCK:
        if msg in _WARNED:
            return
        _WARNED.add(msg)
    print(msg, flush=True)


def _envslug(label: str) -> str:
    """Env-var suffix for a per-arm override:
    'grok-4-1-fast-reasoning' -> 'GROK_4_1_FAST_REASONING'."""
    return receiver_mix.slug(label).upper().replace("-", "_")


def _env_for_label(name: str, label: str, default):
    """Per-arm override with a global fallback: RL_RECV_API_TIMEOUT_<ARM> -> RL_RECV_API_TIMEOUT
    -> default. Returns a float; callers that need an int coerce."""
    v = os.getenv(f"{name}_{_envslug(label)}")
    if v is None:
        return _envf(name, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return _envf(name, default)


def _gateway_url() -> str:
    """Chat-completions endpoint for hosted receivers (RECEIVER_MIX_GATEWAY_URL, read at call time).

    Any OpenAI-compatible endpoint works (e.g. https://openrouter.ai/api/v1/chat/completions, which
    lists deepseek/deepseek-v4-flash), as does a local second receiver
    (http://127.0.0.1:<port>/v1/chat/completions; the launchers resolve the value 'local' to the
    co-served judge for loopback smokes). Default: GATEWAY_URL, the API gateway. There is no
    built-in URL: '' when both are unset, which _gateway_chat() reports as a MIX MISCONFIG."""
    return os.getenv("RECEIVER_MIX_GATEWAY_URL") or os.getenv("GATEWAY_URL") or ""


def _gateway_is_local() -> bool:
    host = _gateway_url().split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    return host in ("127.0.0.1", "localhost", "::1")


def _gateway_key() -> str:
    """Bearer key from the env var named by RECEIVER_MIX_KEY_ENV (default GATEWAY_API_KEY; use
    OPENROUTER_API_KEY for OpenRouter). A loopback endpoint needs no auth, so it gets 'None'."""
    key = os.getenv(os.getenv("RECEIVER_MIX_KEY_ENV", "GATEWAY_API_KEY") or "GATEWAY_API_KEY", "")
    if not key and _gateway_is_local():
        return "None"
    return key


def _api_semaphore(label: str = "api") -> threading.Semaphore:
    """Per-process, per-arm concurrency cap for gateway calls (RL_RECV_API_CONCURRENCY_<ARM>, else
    RL_RECV_API_CONCURRENCY, default 16).

    Per arm because hosted latency differs by an order of magnitude (round-3 prompt:
    DeepSeek-V4-Flash 18.3s, gpt-5-mini 9.8s, grok-4-1-fast-reasoning 13.6s, the rejected grok-4 up
    to 86s), and a shared cap lets the slowest arm block the rest. The rollout runs in N_GPUS
    worker processes plus the reward fallback in the driver, so the global cap per arm is
    ~16x(N_GPUS+1)."""
    with _API_SEM_LOCK:
        sem = _API_SEMS.get(label)
        if sem is None:
            n = int(_env_for_label("RL_RECV_API_CONCURRENCY", label, 16))
            sem = _API_SEMS[label] = threading.Semaphore(max(1, n))
    return sem


# --- Per-model request-body rules --------------------------------------------------------------
# The gateway forwards the body verbatim, so each hosted model's own parameter policy applies.
# Regex on the model id, first match wins. 'drop' removes a key; 'set' forces a value (warning
# once if it differed).
# gpt-5-* / o-series reject 'top_p' (HTTP 400 "Unsupported parameter", reproduced 2/2; 200 without
# it) and any temperature != 1.0.
# reasoning_effort is deliberately unset for gpt-5: full reasoning (a deliberative juror) is the
# chosen receiver configuration. The gateway honours it (minimal: 0 reasoning tokens / 2.2s;
# default: 960 / 9.2s), so setting it changes the experiment.
_API_PARAM_RULES = (
    (r"^(gpt-5|o[1345])", {"drop": ("top_p",), "set": {"temperature": 1.0}}),
    (r".*", {}),
)


def api_body(model_id: str, messages: list, max_tokens: int, *, gateway_url: str = None) -> dict:
    """Request body for every hosted receiver call, shared by _gateway_chat and the launcher
    preflight (scripts/rl_train_sender.slurm) so the preflight ping is byte-identical.

    A preflight with a different body (say, no top_p) can pass while every runtime call gets HTTP
    400, silently training a score-0 arm ('' -> parse_failure -> reward 0 -> zero advantage in a
    receiver-homogeneous group).

    Sends only temperature (RL_RECV_API_TEMPERATURE, 1.0), top_p (RL_RECV_API_TOP_P, 0.95) and
    max_tokens; the Qwen/sglang-specific knobs are never forwarded. Then the per-model rules above,
    then RL_RECV_API_DROP_<ARM>='top_p,frequency_penalty' to drop a parameter without a code change."""
    body = {
        "model": model_id,
        "messages": messages,
        "temperature": _envf("RL_RECV_API_TEMPERATURE", 1.0),
        "top_p": _envf("RL_RECV_API_TOP_P", 0.95),
        "max_tokens": max_tokens,
    }
    for pat, rules in _API_PARAM_RULES:
        if re.match(pat, model_id or "", re.IGNORECASE):
            for k in rules.get("drop", ()):
                body.pop(k, None)
            for k, v in rules.get("set", {}).items():
                if k in body and body[k] != v:
                    _warn_once(
                        f"[receiver_client] {model_id}: forcing {k}={v} (configured {body[k]}) -- "
                        "this arm does NOT honour the shared sampling config; the mixture's arms "
                        "are not sampling-identical")
                body[k] = v
            break
    for k in (x.strip() for x in os.getenv(f"RL_RECV_API_DROP_{_envslug(model_id or '')}", "").split(",")):
        if k:
            body.pop(k, None)
    url = _gateway_url() if gateway_url is None else gateway_url
    # The tier knob is specific to the API gateway. It goes to GATEWAY_URL and to a relay that
    # forwards the body to it under a /gateway/ path (such as
    # http://localhost:18742/gateway/chat/completions), never to another OpenAI-compatible endpoint
    # (OpenRouter, a loopback server).
    if url and (url == os.getenv("GATEWAY_URL") or "/gateway/" in url):
        body["tier"] = os.getenv("GATEWAY_TIER", "base")
    return body


def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _envi(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return int(default)


def receiver_sampling() -> dict:
    """Qwen3.5-35B-A3B recommended sampling, the default for every chat() call. Overridable by
    RL_RECV_TEMPERATURE / _TOP_P / _TOP_K / _MIN_P / _PRESENCE_PENALTY / _REPETITION_PENALTY."""
    return {
        "temperature": _envf("RL_RECV_TEMPERATURE", 1.0),
        "top_p": _envf("RL_RECV_TOP_P", 0.95),
        "top_k": _envi("RL_RECV_TOP_K", 20),
        "min_p": _envf("RL_RECV_MIN_P", 0.0),
        "presence_penalty": _envf("RL_RECV_PRESENCE_PENALTY", 1.5),
        "repetition_penalty": _envf("RL_RECV_REPETITION_PENALTY", 1.0),
    }


def _client() -> OpenAI:
    host, port = local_endpoint()
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="None")


def chat(messages: list, model_id: str = None, max_tokens: int = 8192, max_attempts: int = 4,
         temperature: float = None, top_p: float = None, top_k: int = None, min_p: float = None,
         presence_penalty: float = None, repetition_penalty: float = None,
         enable_thinking: bool = False, seed: int = None) -> str:
    """One blocking chat completion against the local 35B (receiver and judge); '' on failure.

    Sampling args left None use receiver_sampling(). temperature/top_p/presence_penalty go
    top-level; the SGLang-only top_k/min_p/repetition_penalty and enable_thinking ride extra_body.
    """
    s = receiver_sampling()
    temperature = s["temperature"] if temperature is None else temperature
    top_p = s["top_p"] if top_p is None else top_p
    top_k = s["top_k"] if top_k is None else top_k
    min_p = s["min_p"] if min_p is None else min_p
    presence_penalty = s["presence_penalty"] if presence_penalty is None else presence_penalty
    repetition_penalty = s["repetition_penalty"] if repetition_penalty is None else repetition_penalty
    model_id = model_id or _DEFAULT_MODEL
    # sglang forwards chat_template_kwargs to the chat template; the rest are not OpenAI-standard.
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "top_k": int(top_k),
        "min_p": float(min_p),
        "repetition_penalty": float(repetition_penalty),
    }
    seed_kwargs = {}
    if seed is not None:
        seed = int(seed)
        if not 0 <= seed < 2**31 - 1:
            raise ValueError(f"receiver sampling seed must be in [0, 2**31 - 1), got {seed}")
        seed_kwargs["seed"] = seed
    last_err = None
    for _attempt in range(max_attempts):
        try:
            resp = _client().chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=float(temperature),
                top_p=float(top_p),
                presence_penalty=float(presence_penalty),
                max_tokens=max_tokens,
                extra_body=extra_body,
                **seed_kwargs,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 - surface after retries, never crash the reward
            last_err = e
    print(f"[receiver_client] chat failed after {max_attempts} attempts: {last_err}")
    return ""


def _gateway_chat(messages: list, model_id: str = None, label: str = None, max_tokens: int = 8192,
                  max_attempts: int = 4) -> str:
    """One receiver chat completion against the hosted gateway; '' on final failure, like chat().

    Mirrors agents/model/openai_llm.py::GatewayChatModel plus retry/backoff (a shared gateway 429s
    under load). `label` is the mixture arm (the model id in N-way mode, "api" in 2-way) and picks
    the per-arm semaphore and timeout. A missing model id, key or URL (GATEWAY_URL, driver/worker
    env skew) prints a MIX MISCONFIG error so the arm is not mistaken for a gateway outage."""
    model_id = model_id or os.getenv("RECEIVER_MIX_MODEL", "")
    label = label or model_id or "api"
    url = _gateway_url()
    key = _gateway_key()
    if not model_id or not key or not url:
        print("[receiver_client] MIX MISCONFIG: hosted arm requested but "
              f"model id={'set' if model_id else 'MISSING'} / "
              f"bearer key ({os.getenv('RECEIVER_MIX_KEY_ENV', 'GATEWAY_API_KEY')})="
              f"{'set' if key else 'MISSING'} / "
              f"gateway url (GATEWAY_URL, or RECEIVER_MIX_GATEWAY_URL)="
              f"{'set' if url else 'MISSING'} in this process; returning ''", flush=True)
        return ""
    body = api_body(model_id, messages, max_tokens, gateway_url=url)
    timeout = _env_for_label("RL_RECV_API_TIMEOUT", label, 120.0)
    last_err = None
    for attempt in range(max_attempts):
        try:
            with _api_semaphore(label):
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=body, timeout=timeout,
                )
            # A rejected parameter is not transient (retrying burns 2+4+8s and still 400s). Fail
            # fast and log the body: raise_for_status() shows only the status line, which hid the
            # gpt-5-mini top_p rejection.
            if 400 <= resp.status_code < 500 and resp.status_code not in (408, 429):
                _warn_once(f"[receiver_client] {model_id} HTTP {resp.status_code} (non-retryable): "
                           f"{resp.text[:300]} | body keys sent: {sorted(body)}")
                return ""
            resp.raise_for_status()
            return (resp.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:  # noqa: BLE001 - surface after retries, never crash the reward
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(2 ** (attempt + 1))  # 2/4/8s backoff between gateway retries
    print(f"[receiver_client] {model_id} chat failed after {max_attempts} attempts: {last_err}",
          flush=True)
    return ""


def _mix_frac() -> float:
    f = _envf("RECEIVER_MIX_FRAC", 0.5)
    return min(1.0, max(0.0, f))


LOCAL = receiver_mix.LOCAL
API = receiver_mix.API


@functools.lru_cache(maxsize=16)
def _labels_cached(spec: str, model: str) -> tuple:
    if spec:
        return tuple(receiver_mix.labels_of(spec))
    return (API, LOCAL) if model else ()


def mix_labels() -> list:
    """Configured arm labels for this process in canonical order; [] when the mixture is off.

    Memoized on the spec string (called once per reward sample, 256x/step). Prints the arm list
    once per process; that line is the only detector of driver/rollout-worker env skew, under
    which hosted arms silently score 0. This must print exactly one line:

        awk '/^\\[train\\] DIST=/{f=1} f' logs/<job>.out | grep -o 'mix arms=.*' | sort -u

    The awk prefix skips the launcher's rl.check_receiver_mix preflight, which runs both modes in
    its own process and would add a second line.
    """
    labels = list(_labels_cached(os.getenv("RECEIVER_MIX_SPEC", "").strip(),
                                 os.getenv("RECEIVER_MIX_MODEL", "")))
    if labels:
        _warn_once(f"[receiver_client] pid={os.getpid()} mix arms={labels} "
                   f"gateway={_gateway_url() or 'MISSING (set GATEWAY_URL)'}")
    return labels


def nway_labels() -> list:
    """Arms in N-way (RECEIVER_MIX_SPEC) mode only; [] for 2-way or off.

    The reward function emits per-arm reward-extras only for these, so a 2-way run's key set stays
    fixed across resumes."""
    labels = mix_labels()
    return [] if API in labels else labels


def _hash_u(game_id) -> float:
    """Uniform [0,1) from sha1(seed:game_id). Not builtin hash(): PYTHONHASHSEED salting would let
    the rollout worker and the reward fallback disagree about a game."""
    seed = os.getenv("RECEIVER_MIX_SEED", "2026")
    digest = hashlib.sha1(f"{seed}:{game_id}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def resolve_backend(game_id, backend: str = None) -> str:
    """The receiver arm label for one game; 'local' when the mixture is off.

    Precedence:
      1. a stamp from the 10H trainer patch wins;
      2. an unknown stamp raises rather than falling back to local, which would train a
         mislabeled arm. The stamp comes from the same driver process that runs the reward, so
         this fires only on real misconfiguration, and on the reward path it aborts the job;
      3. unstamped rows (validation, eval drivers) use RECEIVER_MIX_PIN if set;
      4. otherwise a per-game sha1 bucket: Bernoulli at RECEIVER_MIX_FRAC (2-way) or cumulative
         weights in spec order (N-way). Stable across processes, resumes and val passes."""
    labels = mix_labels()
    if not labels:
        return LOCAL
    if backend in labels:
        return backend
    if backend == API and API not in labels:
        raise ValueError(
            f"[receiver-mix] 2-way backend 'api' stamped on game {game_id} but RECEIVER_MIX_SPEC "
            f"names {labels}. Either a 2-way checkpoint is being resumed under an N-way env, or "
            "the driver and this process disagree about the mixture configuration.")
    if backend is not None:
        raise ValueError(
            f"[receiver-mix] unknown receiver_backend={backend!r} (game {game_id}); configured "
            f"arms are {labels}. Refusing to fall back to 'local' -- that would train a "
            "mislabeled arm.")
    pin = os.getenv("RECEIVER_MIX_PIN", "").strip()
    if pin:
        if pin not in labels:
            raise ValueError(f"[receiver-mix] RECEIVER_MIX_PIN={pin!r} is not one of {labels}")
        return pin
    u = _hash_u(game_id)
    if API in labels:  # the 2-way RECEIVER_MIX_MODEL form: one hosted arm at RECEIVER_MIX_FRAC
        return API if u < _mix_frac() else LOCAL
    weights = [w for _, w in receiver_mix.parse_spec(os.getenv("RECEIVER_MIX_SPEC", ""))]
    tot, acc = float(sum(weights)), 0.0
    for label, w in zip(labels, weights):
        acc += w / tot
        if u < acc:
            return label
    return labels[-1]  # float-accumulation guard on the last bucket


def receiver_chat(messages: list, game_id=None, backend: str = None,
                  max_tokens: int = 8192, **kwargs) -> str:
    """Receiver-role entry point (training rollout, reward fallback re-query, eval drivers).

    Routes to the local model or a hosted arm per resolve_backend(); judge calls use chat()
    directly. With the mixture off this equals chat(messages, max_tokens=max_tokens, **kwargs)."""
    label = resolve_backend(game_id, backend)
    if label == LOCAL:
        return chat(messages, max_tokens=max_tokens, **kwargs)
    # A hosted arm's model id is its label ('api' means RECEIVER_MIX_MODEL). kwargs are local-only:
    # `model_id` there names the served model (evaluation/rl_rollout.py --receiver-name) and must not
    # override a hosted arm, so all are dropped with a warning.
    if kwargs:
        _warn_once(f"[receiver_client] hosted arm {label!r}: dropping local-only kwargs "
                   f"{sorted(kwargs)} (hosted sampling is RL_RECV_API_* + the per-model rules)")
    model_id = os.getenv("RECEIVER_MIX_MODEL", "") if label == API else label
    return _gateway_chat(messages, model_id=model_id, label=label, max_tokens=max_tokens)
