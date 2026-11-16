"""Standalone eval harness for coding_agent_opd (OpenAI chat transport).

Reuses the exact interaction + grading harness the training loop uses
(``generate.py`` + the ``r2egym`` scaffold/action/observation/prompt/env
modules); only the model transport differs (external OpenAI-compatible
``/v1/chat/completions`` instead of the slime sglang engine + TITO token loop).
"""
