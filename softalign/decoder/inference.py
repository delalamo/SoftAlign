"""
Inference for residue index prediction.

During inference, we decode autoregressively from N-terminus to C-terminus.
The model uses causal masking so that position i can only see index embeddings
from positions j < i (sequentially earlier in N->C order).
"""

import pickle
import numpy as np
import jax
import jax.numpy as jnp

from .model import (
    create_model_fn,
    create_inference_input,
    NUM_INDEX_CLASSES,
    NUM_INPUT_CLASSES,
    MASK_TOKEN_IDX,
)


def create_indices_for_inference(predictions, t, num_input_classes=NUM_INPUT_CLASSES,
                                  mask_token_idx=MASK_TOKEN_IDX):
    """
    Create index embeddings for autoregressive inference at step t.
    """
    return create_inference_input(predictions, t, num_input_classes, mask_token_idx)


def batch_autoregressive_decode(model_fn, params, rng, X, mask, residue_idx,
                                 chain_encoding, greedy=True,
                                 num_classes=NUM_INDEX_CLASSES,
                                 num_input_classes=NUM_INPUT_CLASSES,
                                 mask_token_idx=MASK_TOKEN_IDX):
    """
    Optimized autoregressive decoding for batches using JAX scan.

    Uses causal masking: position i can only see index embeddings from j < i.
    Unassigned positions use mask token (index 129) instead of zeros.
    No monotonicity constraint applied.

    Args:
        model_fn: Haiku transformed model function
        params: Model parameters
        rng: JAX random key
        X: Coordinates [B, L, 4, 3]
        mask: Mask [B, L]
        residue_idx: Sequential residue indices [B, L]
        chain_encoding: Chain encoding [B, L]
        greedy: If True, use greedy decoding
        num_classes: Number of output classes (129)
        num_input_classes: Number of input classes (130)
        mask_token_idx: Index for mask token (129)

    Returns:
        predicted_indices: Predicted indices [B, L]
    """
    B, L = mask.shape

    def decode_step(carry, t):
        """Decode one position."""
        predictions, target_indices_full = carry

        # Get logits (causal mask in decoder handles visibility)
        logits = model_fn.apply(params, rng, X, mask, residue_idx,
                                chain_encoding, target_indices_full)
        logits_t = logits[:, t, :]  # [B, num_classes]

        # Greedy decode
        preds_t = jnp.argmax(logits_t, axis=-1)  # [B]

        # Update predictions
        predictions = predictions.at[:, t].set(preds_t)

        # Update target_indices: put prediction at position t
        new_one_hot = jax.nn.one_hot(preds_t, num_input_classes)  # [B, num_input_classes]
        target_indices_full = target_indices_full.at[:, t, :].set(new_one_hot)

        return (predictions, target_indices_full), preds_t

    # Initialize: all positions start with mask token (index 129)
    predictions = jnp.zeros((B, L), dtype=jnp.int32)
    mask_token_one_hot = jax.nn.one_hot(
        jnp.full((B, L), mask_token_idx, dtype=jnp.int32),
        num_input_classes
    )
    target_indices = mask_token_one_hot

    # Scan over positions
    (final_predictions, _), _ = jax.lax.scan(
        decode_step,
        (predictions, target_indices),
        jnp.arange(L)
    )

    return final_predictions


def predict_residue_indices(model_fn, params, X, mask, residue_idx, chain_encoding,
                            greedy=True):
    """
    High-level function to predict residue indices for a structure.

    Args:
        model_fn: Haiku transformed model
        params: Model parameters
        X: Coordinates [B, L, 4, 3] or [L, 4, 3]
        mask: Mask [B, L] or [L]
        residue_idx: Sequential residue indices [B, L] or [L]
        chain_encoding: Chain encoding [B, L] or [L]
        greedy: Use greedy decoding

    Returns:
        predictions: Predicted IMGT indices [B, L] or [L]
    """
    # Add batch dimension if needed
    single_input = X.ndim == 3
    if single_input:
        X = X[None, ...]
        mask = mask[None, ...]
        residue_idx = residue_idx[None, ...]
        chain_encoding = chain_encoding[None, ...]

    rng = jax.random.PRNGKey(0)
    predictions = batch_autoregressive_decode(
        model_fn, params, rng, X, mask, residue_idx, chain_encoding,
        greedy=greedy
    )

    if single_input:
        predictions = predictions[0]

    return predictions


def postprocess_out_of_range(indices):
    """
    Post-process index 0 predictions to assign actual IMGT positions.

    Index 0 represents "out of IMGT range" which can mean:
    - At N-terminus: positions before IMGT 1 (assigned as 0, -1, -2, ...)
    - At C-terminus: positions after IMGT 128 (assigned as 129, 130, 131, ...)

    Args:
        indices: Array of predicted indices [L] (values 0-128)

    Returns:
        processed: Array with index 0 replaced by actual positions
    """
    indices = np.array(indices)
    L = len(indices)
    processed = indices.copy()

    # Find first non-zero index (first IMGT position)
    first_imgt = None
    for i, idx in enumerate(indices):
        if idx > 0:
            first_imgt = i
            break

    # Find last non-zero index (last IMGT position)
    last_imgt = None
    for i in range(L - 1, -1, -1):
        if indices[i] > 0:
            last_imgt = i
            break

    # Assign N-terminal out-of-range positions (before IMGT 1)
    if first_imgt is not None and first_imgt > 0:
        for i in range(first_imgt):
            # Use negative numbering: 0, -1, -2, ... going backwards
            processed[i] = -(first_imgt - 1 - i)

    # Assign C-terminal out-of-range positions (after IMGT 128)
    if last_imgt is not None and last_imgt < L - 1:
        for i in range(last_imgt + 1, L):
            # Use 129, 130, 131, ... going forwards
            processed[i] = 129 + (i - last_imgt - 1)

    return processed


def add_insertion_codes(indices, postprocess=True):
    """
    Convert predicted indices to IMGT numbers with insertion codes.

    When consecutive residues have the same index, add insertion codes
    (A, B, C, ...) to distinguish them.

    Args:
        indices: Array of predicted indices [L] (values 0-128)
        postprocess: If True, convert index 0 to actual positions first

    Returns:
        imgt_numbers: List of strings like "27", "27A", "27B", etc.
    """
    if postprocess:
        indices = postprocess_out_of_range(indices)

    imgt_numbers = []
    insertion_counts = {}

    for idx in indices:
        idx = int(idx)

        if idx not in insertion_counts:
            # First occurrence: no insertion code
            insertion_counts[idx] = 0
            imgt_numbers.append(str(idx))
        else:
            # Repeated index: add insertion code
            insertion_counts[idx] += 1
            code = chr(ord('A') + insertion_counts[idx] - 1)
            imgt_numbers.append(f"{idx}{code}")

    return imgt_numbers


def load_model(checkpoint_path, hidden_dim=64, num_encoder_layers=3,
               num_decoder_layers=3, k_neighbors=64):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pkl file
        hidden_dim: Hidden dimension
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        k_neighbors: Number of neighbors in structure graph

    Returns:
        model_fn: Haiku transformed model
        params: Loaded parameters
    """
    model_fn = create_model_fn(
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        k_neighbors=k_neighbors,
        dropout=0.0
    )

    with open(checkpoint_path, 'rb') as f:
        params = pickle.load(f)

    return model_fn, params
