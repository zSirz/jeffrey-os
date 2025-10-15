"""
Jeffrey Brain Module - Executive functions and quality control
"""

from .executive_cortex import ArmStats, ExecutiveCortex
from .quality_critic import QualityCritic, ValidationReport

__all__ = ["ExecutiveCortex", "ArmStats", "QualityCritic", "ValidationReport"]
