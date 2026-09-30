"""DynThinker model components (Qwen3-VL + VDPM + Spatial-Dynamic Fusion)."""

from .sdf import SpatialDynamicFusion, is_sdf_layer

__all__ = ["SpatialDynamicFusion", "is_sdf_layer", "load_model_class"]


def load_model_class():
    """Import lazily: the modeling module needs transformers>=4.57."""
    from .modeling_qwen3_vl import Qwen3VLForConditionalGenerationWithVGGT

    return Qwen3VLForConditionalGenerationWithVGGT
