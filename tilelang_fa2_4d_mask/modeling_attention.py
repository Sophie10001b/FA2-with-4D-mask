"""Modeling-code wrappers for replacing an LLM attention implementation."""

from python.modeling_attention import tilelang_attention_forward, tilelang_flash_attention

__all__ = ["tilelang_attention_forward", "tilelang_flash_attention"]
