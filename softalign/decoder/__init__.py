"""
IMGT Residue Index Decoder

Autoregressive MPNN decoder for predicting IMGT residue indices from antibody
backbone coordinates.

Modules:
- model: ResidueIndexDecoder architecture and model creation
- data: Data loading, augmentation, and PDB parsing
- visualization: Metrics tracking and plotting
- training: Loss functions and optimizer utilities
- inference: Autoregressive decoding for prediction
"""

from .model import (
    ResidueIndexDecoder,
    create_model_fn,
    NUM_INDEX_CLASSES,
    NUM_INPUT_CLASSES,
    MASK_TOKEN_IDX,
)

from .data import (
    load_data,
    pad_batch,
    add_coordinate_noise,
    apply_residue_dropout,
    apply_terminus_dropout,
    parse_pdb_for_evaluation,
    remap_imgt_indices,
)

from .inference import (
    batch_autoregressive_decode,
    predict_residue_indices,
    postprocess_out_of_range,
    add_insertion_codes,
    load_model,
)

__all__ = [
    # Model
    'ResidueIndexDecoder',
    'create_model_fn',
    'NUM_INDEX_CLASSES',
    'NUM_INPUT_CLASSES',
    'MASK_TOKEN_IDX',
    # Data
    'load_data',
    'pad_batch',
    'add_coordinate_noise',
    'apply_residue_dropout',
    'apply_terminus_dropout',
    'parse_pdb_for_evaluation',
    'remap_imgt_indices',
    # Inference
    'batch_autoregressive_decode',
    'predict_residue_indices',
    'postprocess_out_of_range',
    'add_insertion_codes',
    'load_model',
]
