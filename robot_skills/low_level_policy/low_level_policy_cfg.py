"""low_level_policy_cfg.py.

The low level policy controller class configuration for skills

Written by Will Solow & Jeff Jewett, 2026
"""

from dataclasses import MISSING

from cfg import configclass


@configclass
class LowLevelPolicyCfg:
    """Configuration of Task Policy."""

    output_dim: int = MISSING
