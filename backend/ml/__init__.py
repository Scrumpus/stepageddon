"""
ML-based step chart generation module.

Provides MLChartGenerator as a drop-in alternative to
ChartGenerationPipeline for generating step charts using
a trained neural network.
"""


def __getattr__(name):
    if name == 'MLChartGenerator':
        from ml.inference import MLChartGenerator
        return MLChartGenerator
    raise AttributeError(f"module 'ml' has no attribute {name}")


__all__ = ['MLChartGenerator']
