"""The native ESPnet learner and its single-host training loop.

``learner.NativeGRPOTrainer`` scores sampled actions with the native model,
turns rewards into GRPO advantages and runs the update; ``local.run`` is the
one-host entry around it. The Ray layout drives the same learner from
``espnet2.rl.miles``.
"""
