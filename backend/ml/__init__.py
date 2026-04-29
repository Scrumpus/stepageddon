"""
ML-based step chart generation module.

Provides MLChartGenerator, the sole entry point for generating
step charts from audio using a trained neural network.
"""


def __getattr__(name):
    if name == 'MLChartGenerator':
        from ml.inference import MLChartGenerator
        return MLChartGenerator
    raise AttributeError(f"module 'ml' has no attribute {name}")


__all__ = ['MLChartGenerator']
