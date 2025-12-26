"""
ResidueIndexDecoder model architecture.

This module contains the autoregressive decoder for predicting IMGT residue
indices from antibody backbone coordinates.

Model Architecture:
- Encoder: ProteinMPNN-style graph neural network (from SoftAlign)
- Decoder: Autoregressive transformer decoder with causal masking

One-Hot Encoding Scheme:
- Output classes: 129 (index 0 = "out of IMGT range", indices 1-128 = IMGT positions)
- Input classes: 130 (indices 0-128 same as output, index 129 = "not yet assigned" mask token)
"""

import jax
import jax.numpy as jnp
import haiku as hk

# Import MPNN from parent softalign package
from .. import MPNN

# Number of residue index classes:
# Index 0 = "out of IMGT range" (before position 1 or after position 128)
# Indices 1-128 = IMGT positions 1-128
NUM_INDEX_CLASSES = 129

# Input classes: output classes + mask token for "not yet assigned"
# Index 0-128: same as output classes
# Index 129: "not yet assigned" / mask token for positions not yet decoded
NUM_INPUT_CLASSES = 130
MASK_TOKEN_IDX = 129  # Index for "not yet assigned" positions


class ResidueIndexDecoder:
    """
    Decoder for predicting residue indices (IMGT numbering).

    Uses the SoftAlign MPNN encoder followed by a custom decoder that
    predicts residue indices in sequential order (N-terminus to C-terminus).

    Output encoding (129 classes):
    - Index 0: "out of IMGT range" (before position 1 or after position 128)
    - Indices 1-128: IMGT positions 1-128

    Input encoding (130 classes):
    - Indices 0-128: same as output
    - Index 129: "not yet assigned" mask token

    Autoregressive decoding with causal masking:
    - Each position i embeds its OWN index (ground truth during training, prediction during inference)
    - For positions j < i: use actual index embedding
    - For positions j >= i: use mask token embedding (index 129)
    - This ensures IDENTICAL behavior during training and inference
    """

    def __init__(self, node_features=64, edge_features=64, hidden_dim=64,
                 num_encoder_layers=3, num_decoder_layers=3,
                 k_neighbors=64, augment_eps=0.0, dropout=0.1,
                 num_classes=NUM_INDEX_CLASSES, num_input_classes=NUM_INPUT_CLASSES):
        """
        Args:
            node_features: Encoder node feature dimension
            edge_features: Encoder edge feature dimension
            hidden_dim: Hidden dimension throughout
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            k_neighbors: Number of neighbors in structure graph
            augment_eps: Backbone noise augmentation
            dropout: Dropout rate
            num_classes: Number of output classes (129: 0=out-of-range, 1-128=IMGT)
            num_input_classes: Number of input classes (130: 0-128 same as output, 129=mask token)
        """
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_input_classes = num_input_classes
        self.num_decoder_layers = num_decoder_layers

        # Encoder (from SoftAlign)
        self.features = MPNN.ProteinFeatures(
            node_features, edge_features,
            top_k=k_neighbors,
            augment_eps=augment_eps
        )
        self.W_e = hk.Linear(hidden_dim, with_bias=True, name='W_e')
        self.encoder_layers = [
            MPNN.EncLayer(hidden_dim, hidden_dim*2, dropout=dropout, name='enc' + str(i))
            for i in range(num_encoder_layers)
        ]

        # Index embedding (for autoregressive decoding)
        # Each position embeds its own index; causal mask controls visibility
        self.index_embed = hk.Linear(hidden_dim, with_bias=True, name='index_embed')

        # Decoder layers
        # Input: [h_VS_expand, h_ES] where h_ES combines edge features with
        # masked index embeddings and encoder node embeddings from neighbors
        # Dimension: hidden_dim + hidden_dim*3 = hidden_dim*4
        self.decoder_layers = [
            MPNN.DecLayer(hidden_dim, hidden_dim*4, dropout=dropout, name='dec' + str(i))
            for i in range(num_decoder_layers)
        ]

        # Output projection
        self.W_out = hk.Linear(num_classes, with_bias=True, name='W_out')

    def encode(self, X, mask, residue_idx, chain_encoding):
        """Encode protein structure using MPNN encoder."""
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding)
        h_V = jnp.zeros((E.shape[0], E.shape[1], E.shape[-1]))
        h_E = self.W_e(E)

        # Encoder attention mask
        mask_attend = MPNN.gather_nodes(jnp.expand_dims(mask, -1), E_idx).squeeze(-1)
        mask_attend = jnp.expand_dims(mask, -1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V, h_E, E_idx

    def decode(self, h_V, h_E, E_idx, target_indices, mask):
        """
        Decode residue indices with causal masking for autoregressive training.

        Each position i can only see index embeddings from positions j < i
        (sequentially earlier in N->C order). For positions j >= i, we use the
        mask token embedding (index 129) - same as during inference.

        Args:
            h_V: Node embeddings from encoder [B, L, hidden_dim]
            h_E: Edge embeddings from encoder [B, L, K, hidden_dim]
            E_idx: Edge indices [B, L, K] - spatial neighbor indices
            target_indices: Ground truth indices (one-hot) [B, L, num_input_classes]
                           Each position has its OWN ground truth embedded
            mask: Mask for valid positions [B, L]

        Returns:
            logits: Residue index logits [B, L, num_classes] (129 classes)
        """
        B, L, K = E_idx.shape

        # Embed target indices at ALL positions
        h_idx = self.index_embed(target_indices)  # [B, L, hidden_dim]

        # Create mask token embedding for "not yet assigned" positions
        # This is used for positions j >= i (positions that i cannot see)
        mask_token_one_hot = jax.nn.one_hot(
            jnp.array(MASK_TOKEN_IDX), self.num_input_classes
        )  # [num_input_classes]
        h_mask = self.index_embed(mask_token_one_hot)  # [hidden_dim]
        # Broadcast to match neighbor dimensions
        h_mask_expanded = jnp.broadcast_to(h_mask, (B, L, K, self.hidden_dim))

        # Build causal mask for N->C sequential decoding
        # order_mask[i, j] = 1 if j < i (position j comes before position i)
        # This is a strictly lower triangular matrix
        positions = jnp.arange(L)
        order_mask = (positions[None, :] < positions[:, None]).astype(jnp.float32)  # [L, L]

        # Gather the causal mask for each position's spatial neighbors
        # For position i with spatial neighbor at position E_idx[i, k],
        # we check if E_idx[i, k] < i (i.e., neighbor comes before i sequentially)
        # mask_causal[b, i, k] = 1 if E_idx[b, i, k] < i, else 0
        i_indices = jnp.broadcast_to(jnp.arange(L)[None, :, None], (B, L, K))
        mask_causal = order_mask[i_indices, E_idx]  # [B, L, K]

        # Combine with validity mask (neighbor must be valid AND come before us)
        mask_valid = MPNN.gather_nodes(jnp.expand_dims(mask, -1), E_idx).squeeze(-1)  # [B, L, K]
        mask_valid = jnp.expand_dims(mask, -1) * mask_valid  # [B, L, K]

        # mask_bw: positions we CAN see (valid AND before us in sequence)
        # mask_fw: positions we CANNOT see index info from (after us or at us)
        mask_bw = mask_valid * mask_causal  # [B, L, K]
        mask_fw = mask_valid * (1.0 - mask_causal)  # [B, L, K]

        # Prepare edge features
        # h_idx_neighbors: index embeddings gathered from spatial neighbors
        h_idx_neighbors = MPNN.gather_nodes(h_idx, E_idx)  # [B, L, K, hidden_dim]

        # For positions we can see (j < i): use actual index embedding
        # For positions we cannot see (j >= i): use mask token embedding
        # This matches inference behavior exactly!
        h_idx_combined = (
            mask_bw[:, :, :, None] * h_idx_neighbors +
            mask_fw[:, :, :, None] * h_mask_expanded
        )  # [B, L, K, hidden_dim]

        # Build full edge features: [h_E, h_idx_combined, h_V_neighbors]
        h_EV_neighbors = MPNN.gather_nodes(h_V, E_idx)  # [B, L, K, hidden_dim]
        h_ES = jnp.concatenate([h_E, h_idx_combined, h_EV_neighbors], -1)  # [B, L, K, hidden_dim*3]

        # For the node update, position i gets its encoder embedding only (not its own h_idx)
        # because it hasn't "predicted" itself yet. We add h_idx from previous positions
        # through message passing.
        h_VS = h_V  # Start with encoder embeddings only

        # Decoder attention mask (just validity, causal masking is in the features)
        mask_attend = mask_valid

        # Run decoder layers
        for layer in self.decoder_layers:
            h_VS = layer(h_VS, h_ES, mask, mask_attend)

        # Output logits
        logits = self.W_out(h_VS)
        return logits

    def __call__(self, X, mask, residue_idx, chain_encoding, target_indices):
        """
        Full forward pass with autoregressive causal masking.

        Args:
            X: Coordinates [B, L, 4, 3]
            mask: Mask [B, L]
            residue_idx: Sequential residue indices [B, L]
            chain_encoding: Chain encoding [B, L]
            target_indices: Ground truth indices (one-hot) [B, L, num_input_classes]
                           Each position has its OWN ground truth for training.
                           Causal masking ensures position i only sees indices from j < i.

        Returns:
            logits: Residue index logits [B, L, num_classes]
        """
        h_V, h_E, E_idx = self.encode(X, mask, residue_idx, chain_encoding)
        logits = self.decode(h_V, h_E, E_idx, target_indices, mask)
        return logits


def create_model_fn(hidden_dim=64, num_encoder_layers=3, num_decoder_layers=3,
                    k_neighbors=64, dropout=0.0):
    """Create the model function for hk.transform."""

    def model_fn(X, mask, residue_idx, chain_encoding, prev_indices):
        model = ResidueIndexDecoder(
            node_features=hidden_dim,
            edge_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            k_neighbors=k_neighbors,
            augment_eps=0.0,
            dropout=dropout,
            num_classes=NUM_INDEX_CLASSES
        )
        return model(X, mask, residue_idx, chain_encoding, prev_indices)

    return hk.transform(model_fn)


def create_target_one_hot(target_indices, num_input_classes=NUM_INPUT_CLASSES):
    """
    Create one-hot encoded target indices for training with causal masking.

    Each position gets its OWN ground truth index as a one-hot vector. The causal
    mask in the decoder controls visibility: position i can only see index embeddings
    from positions j < i (sequentially earlier in N->C order).

    Input dimension is 130 (indices 0-128 for actual values, index 129 for mask token),
    but during training we only use indices 0-128 since causal masking handles visibility.

    Args:
        target_indices: Ground truth indices [B, L] (values 0-128)
        num_input_classes: Number of input classes (130: 0-128 for indices, 129 for mask)

    Returns:
        target_one_hot: One-hot encoded targets [B, L, num_input_classes]
                        Each position has its OWN ground truth index encoded.
    """
    target_one_hot = jax.nn.one_hot(target_indices, num_input_classes)
    return target_one_hot


def create_inference_input(predictions, t, num_input_classes=NUM_INPUT_CLASSES,
                           mask_token_idx=MASK_TOKEN_IDX):
    """
    Create input tensor for autoregressive inference at step t.

    Args:
        predictions: Current predictions [B, L] (positions < t have been filled)
        t: Current position being decoded
        num_input_classes: Number of input classes (130)
        mask_token_idx: Index for mask token (129)

    Returns:
        input_one_hot: One-hot encoded input [B, L, num_input_classes]
                       Positions < t: one-hot of predictions
                       Positions >= t: one-hot of mask token (index 129)
    """
    B, L = predictions.shape

    # Create mask token tensor for unassigned positions
    mask_indices = jnp.full((B, L), mask_token_idx, dtype=jnp.int32)

    # Use predictions for positions < t, mask token for positions >= t
    position_mask = (jnp.arange(L)[None, :] < t)  # [1, L] -> broadcasts to [B, L]
    combined_indices = jnp.where(position_mask, predictions, mask_indices)

    # One-hot encode
    input_one_hot = jax.nn.one_hot(combined_indices, num_input_classes)
    return input_one_hot


# Backward compatibility alias
def create_causal_prev_indices(target_indices, num_input_classes=NUM_INPUT_CLASSES):
    """Backward compatibility wrapper. See create_target_one_hot."""
    return create_target_one_hot(target_indices, num_input_classes)
