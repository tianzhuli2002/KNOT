"""Maintained boundary for visible-event tokenisation and loading.

The mature visible-data implementation was established in the frozen M0 baseline. The
current reconstruction stack imports it through this module so production code does not
depend on the M0 namespace directly. This boundary can later own a physical move of the
loader without changing model/training imports.
"""

from joint_event_reconstruction.m0.data.module import (
    TransformerDataModule,
    _collate_transformer,
    build_token_schema,
)

__all__ = ["TransformerDataModule", "build_token_schema"]
