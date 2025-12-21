import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import argparse
import matplotlib.pyplot as plt
import csv
import time
from typing import Any, Dict


def _unflatten_dict(d: Dict[str, Any], sep: str = "|||") -> Dict[str, Any]:
    """Unflatten a dictionary with separator-joined keys.

    Args:
        d: Flat dictionary with keys like "a|||b|||c".
        sep: Separator used in keys (||| to avoid conflicts with haiku's /).

    Returns:
        Nested dictionary structure.
    """
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def _convert_numpy_to_jax(obj: Any) -> Any:
    """Recursively convert numpy arrays to JAX arrays."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_jax(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return jnp.array(obj)
    else:
        return obj


def load_softalign_params(model_path: str) -> Dict[str, Any]:
    """Load SoftAlign parameters from npz file.

    Args:
        model_path: Path to the model file (with or without .npz extension).

    Returns:
        Dictionary of model parameters.

    Raises:
        FileNotFoundError: If npz file is not found.
    """
    npz_path = model_path + ".npz" if not model_path.endswith(".npz") else model_path
    with open(npz_path, "rb") as f:
        data = dict(np.load(f, allow_pickle=False))
    # Unflatten the dictionary structure and convert to JAX arrays
    params = _unflatten_dict(data)
    params = _convert_numpy_to_jax(params)
    print(f"Loaded model parameters from {npz_path}")
    return params

# Import SoftAlign modules from the cloned repository
# Assuming SoftAlign directory is sibling to this script or in current working dir
# Add SoftAlign directory to sys.path
softalign_path = os.path.join(os.getcwd(), '') # Assuming you are in the SoftAlign directory
if softalign_path not in sys.path:
    sys.path.append(softalign_path)

softalign_code_path = os.path.join(softalign_path, 'softalign')
if softalign_code_path not in sys.path:
    sys.path.append(softalign_code_path)
import ENCODING as enco
import Score_align as score_
import utils
import Input_MPNN as inp
import search
import END_TO_END_MODELS as ete # Needed for pairwise alignment


def load_chain_ids(chain_file_path):
    """Load a dictionary of {pdb_filename: chain_id}"""
    chain_map = {}
    if not chain_file_path or not os.path.exists(chain_file_path):
        return chain_map
    with open(chain_file_path, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if len(row) >= 2:
                pdb_name = row[0].strip()
                chain = row[1].strip()
                chain_map[pdb_name] = chain
    return chain_map

def process_pdb_folder(pdb_folder, chain_file=None, default_chain='A'):
    """Processes PDB files from a folder into the required input format."""
    data = {}
    chain_map = load_chain_ids(chain_file)
    print(f"Processing PDB files from: {pdb_folder}")
    if not os.path.exists(pdb_folder):
        print(f"Error: PDB folder '{pdb_folder}' not found.")
        return data

    for filename in os.listdir(pdb_folder):
        if filename.endswith(".pdb"):
            pdb_path = os.path.join(pdb_folder, filename)
            pdb_key = filename.replace(".pdb", "")
            chain = chain_map.get(pdb_key, default_chain)
            try:
                coords, mask, chain_, res = inp.get_inputs_mpnn(pdb_path, chain=chain)
                data[pdb_key] = (coords, mask, chain_, res)
                # print(f"Processed {filename} using chain {chain}")
            except Exception as e:
                print(f"Error processing {pdb_path}: {e}")
                continue
    print(f"Finished processing {len(data)} PDB files from {pdb_folder}.")
    return data

def main():
    parser = argparse.ArgumentParser(description="Run SoftAlign for protein structural search and optional pairwise visualization.")
    parser.add_argument("--use_scope_database", action="store_true",
                        help="Use precomputed SCOPE database inputs. If false, --pdb_folder_path is required.")
    parser.add_argument("--pdb_folder_path", type=str, default="pdb_files",
                        help="Path to custom PDB folder (required if --use_scope_database is false).")
    parser.add_argument("--chain_ids_file", type=str, default="",
                        help="Path to a CSV file mapping PDB names to chain IDs (e.g., '1abc,A').")
    parser.add_argument("--model_type", type=str, default="Softmax", choices=["Smith-Waterman", "Softmax"],
                        help="Type of SoftAlign model to use.")
    parser.add_argument("--query_id", type=str, default="",
                        help="ID of the query protein for one-vs-all search (e.g., 'd2dixa1'). Leave empty to skip.")
    parser.add_argument("--run_all_vs_all", action="store_true",
                        help="Run a full all-vs-all search across the dataset.")
    parser.add_argument("--pdb1_to_plot_id", type=str, default="",
                        help="First PDB ID for optional pairwise alignment visualization and saving.")
    parser.add_argument("--pdb2_to_plot_id", type=str, default="",
                        help="Second PDB ID for optional pairwise alignment visualization and saving.")
    parser.add_argument("--temperature", type=float, default=1e-4,
                        help="Temperature parameter for soft alignment, used in pairwise plotting.")
    parser.add_argument("--output_dir", type=str, default="./softalign_search_output",
                        help="Directory to save output files (scores, plots, alignment matrix).")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    key = jax.random.PRNGKey(0)

    # --- 1. Load Input Data ---
    dicti_inputs = {}
    if args.use_scope_database:
        print("Downloading SCOPE database inputs...")
        try:
            import gdown
        except ImportError:
            print("gdown not found, installing...")
            os.system("pip install -q gdown")
            import gdown

        gdown.download(id="1DFWcUgPukTxWGPUxaeTM1kNEVNCkRgbO", output=os.path.join(args.output_dir, "dicti_inputs_SCOPE_colab"), quiet=False)
        with open(os.path.join(args.output_dir, "dicti_inputs_SCOPE_colab"), 'rb') as f:
            dicti_inputs = pickle.load(f)
        print("Loaded SCOPE database inputs.")
        # print("Available SCOPE PDB IDs:", dicti_inputs.keys()) # Uncomment to see available IDs
    else:
        if not args.pdb_folder_path:
            print("Error: --pdb_folder_path is required when --use_scope_database is false.")
            sys.exit(1)
        dicti_inputs = process_pdb_folder(args.pdb_folder_path, chain_file=args.chain_ids_file)
        if not dicti_inputs:
            print("No PDB files processed. Exiting.")
            sys.exit(1)
        print("Processed custom PDB folder inputs.")

    # --- 2. Load SoftAlign Model Parameters ---
    params_path_sw = os.path.join(softalign_path, "models", "CONT_SW_05_T_3_1")
    params_path_sft = os.path.join(softalign_path, "models", "CONT_SFT_06_T_3_1")

    if args.model_type == "Smith-Waterman":
        params_path = params_path_sw
    elif args.model_type == "Softmax":
        params_path = params_path_sft
    else:
        raise ValueError("Invalid model type selected.")

    try:
        params = load_softalign_params(params_path)
        print(f"✅ Loaded parameters for {args.model_type} model")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have cloned the SoftAlign repository and downloaded the model weights into the 'models' directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading model parameters: {e}")
        sys.exit(1)

    # --- 3. Prepare data for Encoding ---
    X1s = []
    mask1s = []
    chain1s = []
    res1s = []
    id1s = []
    l1 = []

    for k in dicti_inputs.keys():
        _X1, _mask1, _chain1, _res1 = dicti_inputs[k]
        id1s.append(k)
        X1s.append(_X1[0]) # Assuming batch size 1 for individual inputs
        mask1s.append(_mask1[0])
        chain1s.append(_chain1[0])
        res1s.append(_res1[0])
        l1.append(len(_X1[0]))

    max_len = max(l1) if l1 else 0
    print(f"max_size set to: {max_len}")

    # Convert lists to numpy arrays for batching
    X1s = np.array(X1s, dtype=object)
    mask1s = np.array(mask1s, dtype=object)
    res1s = np.array(res1s, dtype=object)
    chain1s = np.array(chain1s, dtype=object)

    # --- 4. Create Encodings ---
    print("\nStarting structure encoding...")
    encoding_dim = 64
    num_layers = 3
    num_neighbors = 64
    categorical = False
    nb_clusters = 20

    def enco_(x1, node_features=encoding_dim, edge_features=encoding_dim, hidden_dim=encoding_dim,
              num_encoder_layers=num_layers, k_neighbors=num_neighbors, categorical=categorical, nb_clusters=nb_clusters):
        if categorical:
            a = enco.ENCODING_KMEANS_SEQ(node_features, edge_features, hidden_dim, num_encoder_layers, k_neighbors, nb_clusters=nb_clusters)
        else:
            a = enco.ENCODING(node_features, edge_features, hidden_dim, num_encoder_layers, k_neighbors)
        return a(x1)

    ENCO = hk.transform(enco_)
    # JIT compile the encoding function for speed
    enco_fast = jax.jit(ENCO.apply)

    encodings = []
    bs = 10 # Batch size for encoding
    num_samples = len(X1s)

    beg_encoding = time.time()
    for i in range(num_samples // bs):
        batch_X1 = X1s[i * bs:(i + 1) * bs]
        batch_mask1 = mask1s[i * bs:(i + 1) * bs]
        batch_res1 = res1s[i * bs:(i + 1) * bs]
        batch_chain1 = chain1s[i * bs:(i + 1) * bs]

        # Pad the current batch for consistent input shapes to JAX
        # Note: utils.pad_ expects two sets of inputs, here we use the same batch twice for padding purposes
        X_padded, mask_padded, res_padded, chain_padded, _, _, _, _, _ = utils.pad_(
            batch_X1, batch_mask1, batch_res1, batch_chain1,
            batch_X1, batch_mask1, batch_res1, batch_chain1, # Duplicated for padding function signature
            max_len
        )
        input_data = X_padded, mask_padded, res_padded, chain_padded
        encodings_ = enco_fast(params, key, input_data)
        encodings.extend(encodings_)

    if num_samples % bs != 0:
        batch_X1 = X1s[num_samples - num_samples % bs:]
        batch_mask1 = mask1s[num_samples - num_samples % bs:]
        batch_res1 = res1s[num_samples - num_samples % bs:]
        batch_chain1 = chain1s[num_samples - num_samples % bs:]

        X_padded, mask_padded, res_padded, chain_padded, _, _, _, _, _ = utils.pad_(
            batch_X1, batch_mask1, batch_res1, batch_chain1,
            batch_X1, batch_mask1, batch_res1, batch_chain1, # Duplicated for padding function signature
            max_len
        )
        input_data = X_padded, mask_padded, res_padded, chain_padded
        encodings_ = enco_fast(params, key, input_data)
        encodings.extend(encodings_)

    print(f"Encoding finished in {time.time() - beg_encoding:.2f} seconds.")

    dicti_encodings = {}
    for l, k in enumerate(encodings):
        dicti_encodings[id1s[l]] = k[:l1[l], :] # Unpad encodings

    # --- 5. Run One-vs-All Search ---
    if args.query_id:
        print(f"\n--- Running One-vs-All Search for Query ID: {args.query_id} ---")
        enc = dicti_encodings.get(args.query_id)
        if enc is None:
            print(f"Query ID '{args.query_id}' not found in the dataset. Skipping one-vs-all search.")
        else:
            l_query = enc.shape[0]
            l_query_pad = l_query # Use actual length for padding in search

            print(f"Processing single query: {args.query_id} (length={l_query}), using padding {l_query_pad}")
            thresholds = np.arange(100, max_len + 100, 100) # Re-define thresholds
            reusable_target_data = search.setup_target_data(dicti_encodings, dicti_inputs, thresholds)

            try:
                # Removed 'output_folder' argument as it caused an error in your environment
                search.compute_scores_for_query(
                    query_id=args.query_id,
                    target_data=reusable_target_data,
                    model_type=args.model_type,
                    l_query_pad=l_query_pad
                    # Output CSV will likely be saved in the current working directory
                )
                print(f"✅ One-vs-all search scores for {args.query_id} might be saved in the current directory.")
            except Exception as e:
                print(f"Error processing {args.query_id} in one-vs-all search: {e}")

    # --- 6. Run All-vs-All Search ---
    if args.run_all_vs_all:
        print("\n--- Running All-vs-All Search ---")
        thresholds = np.arange(100, max_len + 100, 100)
        reusable_target_data = search.setup_target_data(dicti_encodings, dicti_inputs, thresholds)

        for threshold in thresholds:
            min_len = threshold - 100
            current_max_len = threshold
            l_query_pad = threshold

            print(f"\n=== Queries between {min_len} and {current_max_len} ===")
            compt = 0
            for query_id, enc in dicti_encodings.items():
                l_query = enc.shape[0]
                if min_len <= l_query <= current_max_len:
                    compt += 1
                    print(f"Processing query: {query_id} (length={l_query}), count={compt}")
                    try:
                        # Removed 'output_folder' argument as it caused an error in your environment
                        search.compute_scores_for_query(
                            query_id=query_id,
                            target_data=reusable_target_data,
                            model_type=args.model_type,
                            l_query_pad=l_query_pad
                            # Output CSVs will likely be saved in the current working directory
                        )
                    except Exception as e:
                        print(f"Error processing {query_id} in all-vs-all search: {e}")
        print("\nAll-vs-all search complete. Score CSVs might be saved in the current directory.")

    # --- 7. Optional Pairwise Alignment Visualization and Saving ---
    if args.pdb1_to_plot_id and args.pdb2_to_plot_id:
        print(f"\n--- Performing pairwise alignment for {args.pdb1_to_plot_id} vs {args.pdb2_to_plot_id} ---")
        pdb1_id = args.pdb1_to_plot_id
        pdb2_id = args.pdb2_to_plot_id

        if pdb1_id not in dicti_inputs:
            print(f"Error: PDB ID '{pdb1_id}' not found in the loaded dataset for plotting.")
        elif pdb2_id not in dicti_inputs:
            print(f"Error: PDB ID '{pdb2_id}' not found in the loaded dataset for plotting.")
        else:
            X1, mask1, chain1, res1 = dicti_inputs[pdb1_id]
            X2, mask2, chain2, res2 = dicti_inputs[pdb2_id]

            # Define the END_TO_END model for direct pairwise alignment
            def model_end_to_end(x1, x2, lens, t):
                model = ete.END_TO_END(
                    encoding_dim, encoding_dim, encoding_dim, num_layers,
                    num_neighbors, affine=True, soft_max=False, # Use affine and soft_max=False for typical model
                    dropout=0., augment_eps=0.0)
                return model(x1, x2, lens, t)

            MODEL_ETE = hk.transform(model_end_to_end)

            lens_pair = jnp.array([X1.shape[1], X2.shape[1]])[None, :]

            # Compute soft alignment and similarity matrix for the pair
            soft_aln_pair, sim_matrix_pair, _ = MODEL_ETE.apply(
                params, key, (X1, mask1, chain1, res1), (X2, mask2, chain2, res2), lens_pair, args.temperature
            )

            # Convert JAX arrays to NumPy for plotting and saving
            soft_aln_np = np.asarray(soft_aln_pair[0]) # Assuming batch size 1
            sim_matrix_np = np.asarray(sim_matrix_pair[0]) # Assuming batch size 1

            # Plotting Alignment Matrix
            plt.figure(figsize=(8, 7))
            plt.imshow(soft_aln_np, cmap='viridis', origin='lower',
                       extent=[0, X2.shape[1], 0, X1.shape[1]])
            plt.colorbar(label='Alignment Score')
            plt.xlabel(f'{pdb2_id} Residue Index')
            plt.ylabel(f'{pdb1_id} Residue Index')
            plt.title(f'Soft Alignment Matrix ({pdb1_id} vs {pdb2_id})')
            plt.tight_layout()
            aln_plot_filename = os.path.join(args.output_dir, f'soft_alignment_matrix_{pdb1_id}_vs_{pdb2_id}.png')
            plt.savefig(aln_plot_filename)
            print(f"✅ Saved alignment matrix plot to {aln_plot_filename}")

            # Plotting Similarity Matrix
            plt.figure(figsize=(8, 7))
            plt.imshow(sim_matrix_np, cmap='viridis', origin='lower',
                       extent=[0, X2.shape[1], 0, X1.shape[1]])
            plt.colorbar(label='Similarity Score')
            plt.xlabel(f'{pdb2_id} Residue Index')
            plt.ylabel(f'{pdb1_id} Residue Index')
            plt.title(f'Similarity Matrix ({pdb1_id} vs {pdb2_id})')
            plt.tight_layout()
            sim_plot_filename = os.path.join(args.output_dir, f'similarity_matrix_{pdb1_id}_vs_{pdb2_id}.png')
            plt.savefig(sim_plot_filename)
            print(f"✅ Saved similarity matrix plot to {sim_plot_filename}")
            plt.close('all') # Close plots to free memory

            # Save Alignment Matrix (soft_aln) with PDB names
            alignment_npy_filename = os.path.join(args.output_dir, f'soft_alignment_{pdb1_id}_vs_{pdb2_id}.npy')
            np.save(alignment_npy_filename, soft_aln_np)
            print(f"✅ Saved soft alignment matrix to {alignment_npy_filename}")
    else:
        print("\nTo visualize a specific pairwise alignment, provide --pdb1_to_plot_id and --pdb2_to_plot_id.")

    print("\nProcessing complete. Check the output directory for results.")

if __name__ == "__main__":
    main()