"""Compatibility wrapper for `ml_utils`.

This file keeps the original module name but delegates to the combined
`ml_utils` implementation so existing imports continue to work.
"""
from ml_utils import load_artifacts, predict_single

__all__ = ["load_artifacts", "predict_single"]
