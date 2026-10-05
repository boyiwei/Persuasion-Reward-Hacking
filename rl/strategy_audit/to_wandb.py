"""Backfill the per-step persuasion-strategy distribution into the original RL wandb run.

Reads <RESULTS_ROOT>/<run>/strategy_audit/<step>.jsonl (from rl.strategy_audit.run) and logs each
technique's per-step presence rate (None verdicts ignored) plus legitimacy rollups under
persuasion_strategy/<slug>, resuming the training run by id. The custom x-axis
persuasion_strategy/step keeps wandb from dropping the post-hoc rows (monotonic-_step rule).

Needs network access to wandb. Run id and project are auto-discovered from the wandb run dir whose
config.yaml experiment_name == <run> (override with --run-id / --project).
Validate with --dry-run (logs into a throwaway run), then re-run without it.

  python -m rl.strategy_audit.to_wandb --run sender_qwen3-4B_stubborn_j35B_fakepen0 --dry-run   # validate
  python -m rl.strategy_audit.to_wandb --run sender_qwen3-4B_stubborn_j35B_fakepen0             # backfill
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

from rl.strategy_audit.taxonomy import LEGITIMACIES, by_slug, slugs

_DEFAULT_PROJECT = "persuasion-gym-oldbailey-rl"
# wandb metric namespace: its own top-level group, not under reward_hacking/.
_GROUP = "persuasion_strategy"
_STEP = f"{_GROUP}/step"  # custom x-axis -> immune to wandb's monotonic-_step drop on a resumed run


def _results_root() -> Path:
    env = os.getenv("RESULTS_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "experiments" / "results" / "rl"


def _repo_root() -> Path:
    return Path(os.getenv("REPO_ROOT") or Path(__file__).resolve().parents[1])


def discover_run_local(run_name: str):
    """(run_id, project) from the newest wandb run dir whose config.yaml has
    `experiment_name: <run_name>`; the run id is the dir-name suffix (run-<ts>-<id>). Offline, but
    misses dirs cleaned after sync (then use discover_run_api)."""
    wandb_dir = _repo_root() / "wandb"
    for d in sorted(glob.glob(str(wandb_dir / "*run-*")), reverse=True):
        cfg = Path(d) / "files" / "config.yaml"
        if not cfg.exists():
            continue
        exp = proj = None
        for line in open(cfg, errors="ignore"):
            t = line.strip()
            if t.startswith("experiment_name:"):
                exp = t.split(":", 1)[1].strip().strip("'\"")
            elif t.startswith("project_name:"):
                proj = t.split(":", 1)[1].strip().strip("'\"")
        if exp == run_name:
            return Path(d).name.rsplit("-", 1)[-1], proj
    return None, None


def discover_run_api(run_name: str, entity=None, project=None, project_prefix="persuasion-gym-oldbailey-rl"):
    """(run_id, project) of the newest wandb run named run_name, searching `project` or every project
    starting with project_prefix. Needs network access; (None, None) if unreachable or not
    found."""
    try:
        import wandb
        api = wandb.Api()
    except Exception as exc:  # noqa: BLE001
        print(f"[strategy_to_wandb] wandb API unavailable for discovery: {exc}")
        return None, None
    ent = entity or api.default_entity
    if project:
        projects = [project]
    else:
        try:
            projects = [p.name for p in api.projects(ent) if p.name.startswith(project_prefix)]
        except Exception:  # noqa: BLE001
            projects = [_DEFAULT_PROJECT]
    # Also match the config experiment_name: display names can be reset on a resume.
    best, seen = None, set()
    for proj in projects:
        for filt in ({"display_name": run_name},
                     {"config.trainer/experiment_name": run_name},
                     {"config.experiment_name": run_name}):
            try:
                runs = list(api.runs(f"{ent}/{proj}", filters=filt))
            except Exception:  # noqa: BLE001
                continue
            for r in runs:
                if (proj, r.id) in seen:
                    continue
                seen.add((proj, r.id))
                if best is None or r.created_at > best[0]:
                    best = (r.created_at, r.id, proj)
    return (best[1], best[2]) if best else (None, None)


def per_step_rates(run_name: str):
    """[(step, {slug: rate|None}, n_games)] over <run>/strategy_audit/*.jsonl, ascending by step.
    rate = (# games using the strategy) / (# games with a non-None verdict)."""
    root = _results_root() / run_name / "strategy_audit"
    files = sorted([p for p in root.glob("*.jsonl") if p.stem.isdigit()], key=lambda p: int(p.stem))
    all_slugs = slugs()
    out = []
    for f in files:
        sums = {s: 0 for s in all_slugs}
        cnts = {s: 0 for s in all_slugs}
        n_games = 0
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                vec = (json.loads(line).get("vector") or {})
                n_games += 1
                for s in all_slugs:
                    v = vec.get(s)
                    if v is not None:
                        cnts[s] += 1
                        sums[s] += 1 if v >= 0.5 else 0
        rates = {s: (sums[s] / cnts[s]) if cnts[s] else None for s in all_slugs}
        out.append((int(f.stem), rates, n_games))
    return out


def _rollups(rates: dict) -> dict:
    """Mean #strategies-per-game by legitimacy (= sum of presence rates) + fraction legit."""
    legit = by_slug()
    by = {lv: 0.0 for lv in LEGITIMACIES}
    for slug, r in rates.items():
        if r is not None:
            by[legit[slug]["legitimacy"]] += r
    total = sum(by.values())
    row = {f"{_GROUP}/_n_{lv}_mean": by[lv] for lv in LEGITIMACIES}
    row[f"{_GROUP}/_frac_legit_mean"] = (by["legit"] / total) if total > 0 else None
    return row


def log_to_wandb(steps_data, run_id, project, entity=None, dry_run=False):
    import wandb
    init_kwargs = {"project": project, "entity": entity}
    if dry_run:
        init_kwargs.update(name=f"{run_id}__strategy_audit_DRYRUN", tags=["strategy_audit", "dryrun"])
    else:
        init_kwargs.update(id=run_id, resume="must")
    run = wandb.init(**init_kwargs)
    wandb.define_metric(_STEP)
    wandb.define_metric(f"{_GROUP}/*", step_metric=_STEP)
    for step, rates, n_games in steps_data:
        row = {_STEP: step, f"{_GROUP}/_n_games": n_games}
        for slug, r in rates.items():
            if r is not None:
                row[f"{_GROUP}/{slug}"] = r
        row.update({k: v for k, v in _rollups(rates).items() if v is not None})
        wandb.log(row)
    run.finish()
    return run.id


def main(argv=None):
    p = argparse.ArgumentParser(description="Backfill per-step strategy distribution into the RL wandb run.")
    p.add_argument("--run", required=True, help="run dir name under experiments/results/rl/")
    p.add_argument("--run-id", default=None, help="wandb run id to resume (default: auto-discover)")
    p.add_argument("--project", default=None, help="wandb project (default: auto-discover, else %(default)s)")
    p.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    p.add_argument("--dry-run", action="store_true", help="log into a fresh throwaway run, not the real one")
    a = p.parse_args(argv)

    steps_data = per_step_rates(a.run)
    if not steps_data:
        sys.exit(f"[strategy_to_wandb] no strategy_audit/*.jsonl for run {a.run} "
                 f"(under {_results_root() / a.run / 'strategy_audit'}) -- run rl.strategy_audit.run first")

    run_id, project = a.run_id, a.project
    if not run_id:  # local config.yaml scan first (fast, offline)
        run_id, disc_proj = discover_run_local(a.run)
        project = project or disc_proj
    if not run_id and not a.dry_run:  # wandb API fallback (handles cleaned dirs / dated projects)
        run_id, api_proj = discover_run_api(a.run, entity=a.entity, project=a.project)
        project = project or api_proj
    project = project or _DEFAULT_PROJECT
    if a.dry_run and not run_id:
        run_id = a.run  # a label is enough for the throwaway run
    if not run_id:
        sys.exit(f"[strategy_to_wandb] could not discover a wandb run id for {a.run}; pass --run-id "
                 f"(and --project). Tip: wandb API search needs network access.")

    print(f"[strategy_to_wandb] run={a.run} steps={len(steps_data)} run_id={run_id} project={project} "
          f"entity={a.entity} dry_run={a.dry_run}", flush=True)
    try:
        logged = log_to_wandb(steps_data, run_id, project, entity=a.entity, dry_run=a.dry_run)
        print(f"[strategy_to_wandb] done -> wandb run {logged} ({project})", flush=True)
    except Exception as exc:  # noqa: BLE001 -- never crash; the JSONL vectors are the source of truth
        print(f"[strategy_to_wandb] wandb logging failed ({type(exc).__name__}): {exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
