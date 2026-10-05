"""SFT stage of the SFT-then-RL sender curriculum (Old Bailey).

Modules:
  sft_dataset -- PersuasionMultiTurnSFTDataset for verl's fsdp_sft_trainer: verl's
                 MultiTurnSFTDataset + the Qwen3 thinking-template fix that makes
                 incremental tokenization prefix-consistent AND token-exact with the
                 GRPO rollout stream.

The corpora it trains on are built by datasets/old_bailey/sft/build_sft_dataset.py (demonstration
SFT) and datasets/old_bailey/aux_loss/build_auxce_sft_dataset.py (the aux-CE probe corpus).

Depends on rl/* (taxonomy, reward_function format predicates, persuasion_interaction templates);
the dependency direction is sft -> rl only.
"""
