"""Shared utilities for the coding_agent_opd recipe.

- ``rollout``: env-driven config + rollout helpers (scaffold/budget, pod boot,
  ds validation, transport guards, sample/abort bookkeeping) supporting
  ``generate.py``'s core loop; also reused by ``eval/eval_chat.py``.
- ``filters``: the ``drop_aborted_groups`` dynamic-sampling filter.
"""
