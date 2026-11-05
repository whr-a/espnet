"""The adapter that runs native ESPnet training inside the pinned Miles loop.

Only this package reads or writes Miles module globals: ``driver`` binds
ESPnet's factories into ``train_async.train``'s namespace and ``components``
owns Miles' object store for one orchestration. ``actor`` is the training
worker that implements Miles' ``TrainRayActor`` contract; ``failure``
preserves its state when a step fails before the optimizer runs.
"""
