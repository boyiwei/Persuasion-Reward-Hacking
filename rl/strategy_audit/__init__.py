"""Offline persuasion-strategy audit of RL sender rollouts.

Submodules:
  taxonomy  -- the 42-strategy taxonomy (single source of truth)
  audit     -- one binary LLM-judge call per (strategy, argument)
  run       -- offline driver: rollouts -> per-game 42-vectors (python -m rl.strategy_audit.run)
  to_wandb  -- backfill of the per-step distribution into the original wandb run
"""
