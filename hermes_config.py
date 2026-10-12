# -*- coding: utf-8 -*-
"""
hermes_config.py — single source of truth for configuration.

Every script imports from here; never copy paths or ports into other files.
A copied value silently rots when reality changes — and it fails silently,
which is worse. Edit this file to fit your deployment; override with
environment variables where noted.
"""
import os

# Root of this repo. All default paths derive from it.
HOME = os.path.dirname(os.path.abspath(__file__))

# Local Ollama (bge-m3 embeddings). Override with OLLAMA_PORT env var.
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))
OLLAMA_URL = f"http://localhost:{OLLAMA_PORT}/api/embeddings"

# The memory file. Override with SEMANTIC_MEMORIES env var.
SEMANTIC_MEMORIES = os.environ.get(
    "SEMANTIC_MEMORIES", os.path.join(HOME, "semantic_memories.jsonl")
)
