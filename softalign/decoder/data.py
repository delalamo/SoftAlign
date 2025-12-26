"""
Data loading, augmentation, and preprocessing for IMGT residue index training.

This module handles:
- Loading training/validation data from pickle files
- Batch padding and collation
- Data augmentation (coordinate noise, residue dropout, terminus dropout)
- PDB parsing for evaluation
"""

import pickle
import numpy as np
import jax
import jax.numpy as jnp


def remap_imgt_indices(imgt_indices):
    """
    Remap IMGT indices from old scheme (0-129) to new scheme (0-128).

    Old scheme:
    - 0: residues before IMGT position 1
    - 1-128: IMGT positions 1-128
    - 129: residues after IMGT position 128

    New scheme:
    - 0: "out of IMGT range" (combines old 0 and 129)
    - 1-128: IMGT positions 1-128 (unchanged)

    Args:
        imgt_indices: Array of indices in old scheme (values 0-129)

    Returns:
        remapped: Array of indices in new scheme (values 0-128)
    """
    # Map 129 -> 0, keep everything else the same
    remapped = np.where(imgt_indices == 129, 0, imgt_indices)
    return remapped


def load_data(dicti_path, imgt_path, max_samples=None):
    """
    Load training data.

    Automatically remaps IMGT indices from old scheme (0-129) to new scheme (0-128):
    - Old index 129 (after IMGT 128) is remapped to 0 (out-of-range)
    - Indices 1-128 remain unchanged

    Args:
        dicti_path: Path to dicti_inputs pickle
        imgt_path: Path to imgt_data pickle
        max_samples: Maximum number of samples to load

    Returns:
        data: List of dicts with keys: X, mask, chain, res, imgt
    """
    with open(dicti_path, 'rb') as f:
        dicti_inputs = pickle.load(f)

    with open(imgt_path, 'rb') as f:
        imgt_data = pickle.load(f)

    data = []
    for protein_id in list(dicti_inputs.keys())[:max_samples]:
        if protein_id not in imgt_data:
            continue

        X, mask, chain, res = dicti_inputs[protein_id]
        imgt_indices = imgt_data[protein_id]

        # Ensure lengths match
        L = X.shape[1]
        if len(imgt_indices) != L:
            continue

        # Remap IMGT indices: 129 -> 0 (both represent "out of IMGT range")
        imgt_indices = remap_imgt_indices(np.array(imgt_indices))

        data.append({
            'X': X[0],  # Remove batch dim
            'mask': mask[0],
            'chain': chain[0],
            'res': res[0],
            'imgt': imgt_indices
        })

    return data


def pad_batch(batch, max_len):
    """
    Pad a batch of samples to the same length.

    Args:
        batch: List of data dictionaries
        max_len: Maximum sequence length

    Returns:
        Padded arrays: X, mask, chain, res, imgt, lens
    """
    B = len(batch)

    X = np.zeros((B, max_len, 4, 3), dtype=np.float32)
    mask = np.zeros((B, max_len), dtype=np.float32)
    chain = np.zeros((B, max_len), dtype=np.int32)
    res = np.zeros((B, max_len), dtype=np.int32)
    imgt = np.zeros((B, max_len), dtype=np.int32)
    lens = np.zeros(B, dtype=np.int32)

    for i, sample in enumerate(batch):
        L = len(sample['X'])
        L = min(L, max_len)
        lens[i] = L

        X[i, :L] = sample['X'][:L]
        mask[i, :L] = sample['mask'][:L]
        chain[i, :L] = sample['chain'][:L]
        res[i, :L] = sample['res'][:L]
        imgt[i, :L] = sample['imgt'][:L]

    return X, mask, chain, res, imgt, lens


def add_coordinate_noise(X, key, noise_std=0.2):
    """
    Add Gaussian noise to atomic coordinates for data augmentation.

    Args:
        X: Backbone coordinates [B, L, 4, 3]
        key: JAX random key
        noise_std: Standard deviation of Gaussian noise (default 0.2 Angstroms)

    Returns:
        X_noisy: Coordinates with added noise [B, L, 4, 3]
    """
    noise = jax.random.normal(key, shape=X.shape) * noise_std
    return X + noise


def apply_residue_dropout(X, mask, res, chain, imgt, rng_key, dropout_frac=0.01):
    """
    Randomly remove 1-10 residues from a fraction of samples.

    This simulates missing residues in structures (e.g., disordered regions,
    crystallographic gaps). The IMGT indices of remaining residues are preserved.

    Args:
        X: Backbone coordinates [B, L, 4, 3]
        mask: Valid position mask [B, L]
        res: Sequential residue indices [B, L]
        chain: Chain encoding [B, L]
        imgt: IMGT indices [B, L]
        rng_key: JAX random key
        dropout_frac: Fraction of samples to augment (default 0.01 = 1%)

    Returns:
        Tuple of (X, mask, res, chain, imgt) with residues removed from some samples
    """
    B, L = mask.shape

    # Split keys for different random operations
    key1, key2, key3 = jax.random.split(rng_key, 3)

    # Decide which samples to augment
    augment_mask = jax.random.uniform(key1, (B,)) < dropout_frac

    # For each sample, decide how many residues to remove (1-10)
    n_remove = jax.random.randint(key2, (B,), 1, 11)  # 1 to 10 inclusive

    # Process each sample
    X_out = X.copy()
    mask_out = mask.copy()
    res_out = res.copy()
    chain_out = chain.copy()
    imgt_out = imgt.copy()

    for b in range(B):
        if not augment_mask[b]:
            continue

        # Get valid positions for this sample
        valid_positions = jnp.where(mask[b] > 0)[0]
        n_valid = len(valid_positions)

        if n_valid <= 10:
            # Too few residues, skip
            continue

        # Number of residues to remove (capped at n_valid - 1 to keep at least 1)
        n_to_remove = min(int(n_remove[b]), n_valid - 1)

        # Randomly select positions to remove
        key3, subkey = jax.random.split(key3)
        remove_indices = jax.random.choice(
            subkey, valid_positions, shape=(n_to_remove,), replace=False
        )

        # Create a mask of positions to keep
        keep_mask = jnp.ones(L, dtype=bool)
        keep_mask = keep_mask.at[remove_indices].set(False)

        # Get indices of positions to keep (in order)
        keep_positions = jnp.where(keep_mask & (mask[b] > 0))[0]

        # Shift data: move kept positions to the front
        X_new = jnp.zeros_like(X[b])
        mask_new = jnp.zeros_like(mask[b])
        res_new = jnp.zeros_like(res[b])
        chain_new = jnp.zeros_like(chain[b])
        imgt_new = jnp.zeros_like(imgt[b])

        for i, pos in enumerate(keep_positions):
            X_new = X_new.at[i].set(X[b, pos])
            mask_new = mask_new.at[i].set(mask[b, pos])
            res_new = res_new.at[i].set(i)  # Re-index sequentially
            chain_new = chain_new.at[i].set(chain[b, pos])
            imgt_new = imgt_new.at[i].set(imgt[b, pos])  # Keep IMGT indices!

        X_out = X_out.at[b].set(X_new)
        mask_out = mask_out.at[b].set(mask_new)
        res_out = res_out.at[b].set(res_new)
        chain_out = chain_out.at[b].set(chain_new)
        imgt_out = imgt_out.at[b].set(imgt_new)

    return X_out, mask_out, res_out, chain_out, imgt_out


def apply_terminus_dropout(X, mask, res, chain, imgt, rng_key, dropout_frac=0.01):
    """
    Randomly remove 1-7 residues from N or C terminus of a fraction of samples.

    This simulates truncated structures (e.g., disordered termini, expression
    constructs with shortened ends). The IMGT indices of remaining residues
    are preserved.

    Args:
        X: Backbone coordinates [B, L, 4, 3]
        mask: Valid position mask [B, L]
        res: Sequential residue indices [B, L]
        chain: Chain encoding [B, L]
        imgt: IMGT indices [B, L]
        rng_key: JAX random key
        dropout_frac: Fraction of samples to augment (default 0.01 = 1%)

    Returns:
        Tuple of (X, mask, res, chain, imgt) with terminus residues removed from some samples
    """
    B, L = mask.shape

    # Split keys for different random operations
    key1, key2, key3 = jax.random.split(rng_key, 3)

    # Decide which samples to augment
    augment_mask = jax.random.uniform(key1, (B,)) < dropout_frac

    # For each sample, decide how many residues to remove (1-7)
    n_remove = jax.random.randint(key2, (B,), 1, 8)  # 1 to 7 inclusive

    # For each sample, decide which terminus (0 = N-terminus, 1 = C-terminus)
    terminus_choice = jax.random.randint(key3, (B,), 0, 2)

    # Process each sample
    X_out = X.copy()
    mask_out = mask.copy()
    res_out = res.copy()
    chain_out = chain.copy()
    imgt_out = imgt.copy()

    for b in range(B):
        if not augment_mask[b]:
            continue

        # Get valid positions for this sample
        valid_positions = jnp.where(mask[b] > 0)[0]
        n_valid = len(valid_positions)

        if n_valid <= 8:
            # Too few residues, skip
            continue

        # Number of residues to remove (capped at n_valid - 1 to keep at least 1)
        n_to_remove = min(int(n_remove[b]), n_valid - 1)

        # Determine which positions to remove based on terminus choice
        if terminus_choice[b] == 0:
            # N-terminus: remove first n_to_remove valid positions
            remove_indices = valid_positions[:n_to_remove]
        else:
            # C-terminus: remove last n_to_remove valid positions
            remove_indices = valid_positions[-n_to_remove:]

        # Create a mask of positions to keep
        keep_mask = jnp.ones(L, dtype=bool)
        keep_mask = keep_mask.at[remove_indices].set(False)

        # Get indices of positions to keep (in order)
        keep_positions = jnp.where(keep_mask & (mask[b] > 0))[0]

        # Shift data: move kept positions to the front
        X_new = jnp.zeros_like(X[b])
        mask_new = jnp.zeros_like(mask[b])
        res_new = jnp.zeros_like(res[b])
        chain_new = jnp.zeros_like(chain[b])
        imgt_new = jnp.zeros_like(imgt[b])

        for i, pos in enumerate(keep_positions):
            X_new = X_new.at[i].set(X[b, pos])
            mask_new = mask_new.at[i].set(mask[b, pos])
            res_new = res_new.at[i].set(i)  # Re-index sequentially
            chain_new = chain_new.at[i].set(chain[b, pos])
            imgt_new = imgt_new.at[i].set(imgt[b, pos])  # Keep IMGT indices!

        X_out = X_out.at[b].set(X_new)
        mask_out = mask_out.at[b].set(mask_new)
        res_out = res_out.at[b].set(res_new)
        chain_out = chain_out.at[b].set(chain_new)
        imgt_out = imgt_out.at[b].set(imgt_new)

    return X_out, mask_out, res_out, chain_out, imgt_out


def parse_pdb_for_evaluation(pdb_path):
    """
    Parse a PDB file to extract backbone coordinates and IMGT indices.

    Args:
        pdb_path: Path to PDB file with IMGT numbering

    Returns:
        X: Backbone coordinates [L, 4, 3] (N, CA, C, O)
        imgt_indices: IMGT indices [L] (values 1-128)
        residue_ids: List of (resnum, insertion_code) tuples
    """
    backbone_atoms = ['N', 'CA', 'C', 'O']

    # Read all ATOM lines
    atoms = {}  # (resnum, icode) -> {atom_name: (x, y, z)}

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name in backbone_atoms:
                    resnum = int(line[22:26].strip())
                    icode = line[26].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    key = (resnum, icode)
                    if key not in atoms:
                        atoms[key] = {}
                    atoms[key][atom_name] = (x, y, z)

    # Sort by residue number and insertion code
    residue_ids = sorted(atoms.keys(), key=lambda x: (x[0], x[1]))

    # Build coordinate array
    X = []
    imgt_indices = []

    for resnum, icode in residue_ids:
        atom_dict = atoms[(resnum, icode)]
        if all(a in atom_dict for a in backbone_atoms):
            coords = [atom_dict[a] for a in backbone_atoms]
            X.append(coords)
            # IMGT index is the residue number (1-128)
            imgt_indices.append(resnum)

    X = np.array(X, dtype=np.float32)  # [L, 4, 3]
    imgt_indices = np.array(imgt_indices, dtype=np.int32)

    return X, imgt_indices, residue_ids


def get_region(imgt_pos):
    """
    Get region name for an IMGT position.

    Args:
        imgt_pos: IMGT position (1-128)

    Returns:
        Region name: 'FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4', or 'Other'
    """
    if 1 <= imgt_pos <= 26:
        return 'FR1'
    elif 27 <= imgt_pos <= 38:
        return 'CDR1'
    elif 39 <= imgt_pos <= 55:
        return 'FR2'
    elif 56 <= imgt_pos <= 65:
        return 'CDR2'
    elif 66 <= imgt_pos <= 104:
        return 'FR3'
    elif 105 <= imgt_pos <= 117:
        return 'CDR3'
    elif 118 <= imgt_pos <= 128:
        return 'FR4'
    return 'Other'
