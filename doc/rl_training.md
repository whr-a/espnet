# Optional speech reinforcement learning

The canonical [RL guide](../espnet2/rl/README.md) covers model ownership,
ESPnet2 placement, public preparation, the single training entry, numerical
contracts and held-out evaluation. Follow its
[environment setup](../espnet2/rl/ENVIRONMENT.md) before using a reviewed
integration checkout.

Main profiles use one H100 for collocation or two H100s for disaggregated
asynchronous execution, with32 prompts × 8 candidates per update. OWSM WER,
Bagpiper QA/TTS and OpusLM retain model-specific preparation/reward adapters
inside ESPnet. Existing supervised stages keep their behavior. Templates and
CPU checks do not establish every model's GPU fit or quality improvement.
