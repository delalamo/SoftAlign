"""
Training utilities: loss functions, weight decay, and optimizer helpers.

This module provides:
- Cross-entropy loss with masking
- Weight decay computation (both L2 and decoupled AdamW style)
- Pretrained encoder weight loading
"""

import pickle
import jax
import jax.numpy as jnp


def cross_entropy_loss(logits, targets, mask):
    """
    Compute cross entropy loss with masking.

    Args:
        logits: Predicted logits [B, L, num_classes]
        targets: Ground truth indices [B, L]
        mask: Valid position mask [B, L]

    Returns:
        loss: Scalar loss value
        accuracy: Accuracy on valid positions
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    ).squeeze(-1)

    masked_log_probs = target_log_probs * mask
    loss = -jnp.sum(masked_log_probs) / (jnp.sum(mask) + 1e-8)

    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == targets) * mask
    accuracy = jnp.sum(correct) / (jnp.sum(mask) + 1e-8)

    return loss, accuracy


def load_pretrained_encoder(params, pretrained_path):
    """
    Load pretrained SoftAlign encoder weights.

    Only loads encoder-related parameters, leaving decoder randomly initialized.

    Args:
        params: Initialized model parameters
        pretrained_path: Path to pretrained SoftAlign weights

    Returns:
        Tuple of (updated_params, pretrained_encoder_params)
    """
    with open(pretrained_path, 'rb') as f:
        pretrained = pickle.load(f)

    new_params = {}
    pretrained_encoder_params = {}
    encoder_count = 0
    decoder_count = 0

    for key in params.keys():
        if key in pretrained:
            new_params[key] = pretrained[key]
            pretrained_encoder_params[key] = pretrained[key]
            encoder_count += 1
            print(f"  Loaded (encoder): {key}")
        else:
            new_params[key] = params[key]
            decoder_count += 1
            print(f"  Random (decoder): {key}")

    print(f"\n  Summary: {encoder_count} encoder params, {decoder_count} decoder params")
    return new_params, pretrained_encoder_params


def compute_weight_decay_loss(params, pretrained_encoder_params, weight_decay):
    """
    Compute weight decay loss with different targets (for Adam optimizer).

    - Encoder params: L2 regularization toward pretrained weights
    - Decoder params: L2 regularization toward zero
    """
    total_loss = 0.0

    for key in params.keys():
        param_dict = params[key]

        if key in pretrained_encoder_params:
            pretrained_dict = pretrained_encoder_params[key]
            for subkey in param_dict.keys():
                if subkey in pretrained_dict:
                    diff = param_dict[subkey] - pretrained_dict[subkey]
                    total_loss += jnp.sum(diff ** 2)
        else:
            for subkey in param_dict.keys():
                total_loss += jnp.sum(param_dict[subkey] ** 2)

    return weight_decay * total_loss


def compute_decoupled_weight_decay(params, pretrained_encoder_params, weight_decay):
    """
    Compute decoupled weight decay updates for AdamW optimizer.

    - Encoder params: decay toward pretrained weights
    - Decoder params: decay toward zero
    """
    decay_updates = {}

    for key in params.keys():
        param_dict = params[key]
        decay_updates[key] = {}

        if key in pretrained_encoder_params:
            pretrained_dict = pretrained_encoder_params[key]
            for subkey in param_dict.keys():
                if subkey in pretrained_dict:
                    decay_updates[key][subkey] = weight_decay * (
                        param_dict[subkey] - pretrained_dict[subkey]
                    )
                else:
                    decay_updates[key][subkey] = weight_decay * param_dict[subkey]
        else:
            for subkey in param_dict.keys():
                decay_updates[key][subkey] = weight_decay * param_dict[subkey]

    return decay_updates


def apply_decoupled_weight_decay(params, pretrained_encoder_params, weight_decay, learning_rate):
    """Apply decoupled weight decay for AdamW (after gradient update)."""
    decay_updates = compute_decoupled_weight_decay(
        params, pretrained_encoder_params, weight_decay
    )
    new_params = {}
    total_decay_sq = 0.0
    for key in params.keys():
        new_params[key] = {}
        for subkey in params[key].keys():
            decay_term = learning_rate * decay_updates[key][subkey]
            new_params[key][subkey] = params[key][subkey] - decay_term
            total_decay_sq += float(jnp.sum(decay_updates[key][subkey] ** 2))

    wd_equivalent = 0.5 * total_decay_sq / weight_decay if weight_decay > 0 else 0.0
    return new_params, wd_equivalent
