"""
COBRA Model Integration Module

This module provides high-level integration with COBRA models:
- compress_cobra_model(): Main compression function
- CompressionResult: Complete compression results
- CompressionConverter: Bidirectional expression transformation
- Preprocessing utilities for model preparation

Provides the clean API specified in API_SPECIFICATION.md
"""

from .compressor import compress_cobra_model, preprocess_cobra_model
from .result import CompressionResult
from .converter import CompressionConverter, create_compression_converter

__all__ = [
    'compress_cobra_model',
    'preprocess_cobra_model', 
    'CompressionResult',
    'CompressionConverter',
    'create_compression_converter',
]