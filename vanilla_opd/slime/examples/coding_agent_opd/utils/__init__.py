"""Shared utilities for the coding_agent_opd recipe.

- ``rollout``: env-driven config + rollout helpers (scaffold/budget, pod boot,
  ds validation, transport guards, sample/abort bookkeeping) supporting
  ``generate.py``'s core loop; also reused by the sibling package's
  ``eval/eval_chat.py`` (``../offline_opd``).
- ``filters``: the ``drop_aborted_groups`` dynamic-sampling filter.
"""
