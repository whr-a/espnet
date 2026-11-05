"""Rollout: the families' inference engines in their own processes.

``engine`` is the interface a model family implements; ``worker`` is the
process that holds one engine for a whole run and ``process`` its client;
``local`` runs one engine process per rollout GPU on this host and
publishes each policy to them, by disk export or through ``weights``.
"""
