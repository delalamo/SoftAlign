import functools

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from . import utils

Gelu = functools.partial(jax.nn.gelu, approximate=False)

class SafeKey:
    """Safety wrapper for PRNG keys."""

    def __init__(self, key):
        self._key = key
        self._used = False

    def _assert_not_used(self):
        if self._used:
            raise RuntimeError('Random key has been used previously.')

    def get(self):
        self._assert_not_used()
        self._used = True
        return self._key

    def split(self, num_keys=2):
        self._assert_not_used()
        self._used = True
        new_keys = jax.random.split(self._key, num_keys)
        return jax.tree_util.tree_map(SafeKey, tuple(new_keys))

    def duplicate(self, num_keys=2):
        self._assert_not_used()
        self._used = True
        return tuple(SafeKey(self._key) for _ in range(num_keys))

def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    #neighbors = jnp.expand_dims(neighbor_idx, -1).tile([1, 1, 1, edges.shape[-1]])
    neighbors = jnp.tile(jnp.expand_dims(neighbor_idx, -1),[1, 1, 1, edges.shape[-1]])
    edge_features = jnp.take_along_axis(edges, neighbors, 2)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape([neighbor_idx.shape[0], -1])
    #neighbors_flat = jnp.expand_dims(neighbors_flat, -1).tile([1, 1, nodes.shape[2]])
    neighbors_flat = jnp.tile(jnp.expand_dims(neighbors_flat, -1),[1, 1, nodes.shape[2]])
    # Gather and re-pack
    neighbor_features = jnp.take_along_axis(nodes, neighbors_flat, 1)
    neighbor_features = neighbor_features.reshape(list(neighbor_idx.shape[:3]) + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    #idx_flat = jnp.expand_dims(neighbor_idx, -1).tile([1, 1, nodes.shape[2]])
    idx_flat = jnp.tile(jnp.expand_dims(neighbor_idx, -1),[1, 1, nodes.shape[2]])
    neighbor_features = jnp.take_along_axis(nodes, idx_flat, 1)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = jnp.concatenate([h_neighbors, h_nodes], -1)
    return h_nn

class PositionalEncodings(hk.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = hk.Linear(num_embeddings, name='embedding_linear')

    def __call__(self, offset, mask):
        d = jnp.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature) * mask + \
            (1 - mask) * (2*self.max_relative_feature + 1)
        d_onehot = jax.nn.one_hot(d, 2*self.max_relative_feature + 1 + 1)
        E = self.linear(jax.lax.convert_element_type(d_onehot, jnp.float32))
        return E
class ProteinFeatures(hk.Module):
    def __init__(self, edge_features, node_features,
                 num_positional_embeddings=16,
                 num_rbf=16, top_k=30,
                 augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = hk.Linear(edge_features, with_bias=False, name='edge_embedding')
        self.norm_edges = hk.LayerNorm(-1, create_scale=True, create_offset=True, name='norm_edges')

        #self.safe_key = SafeKey(hk.next_rng_key())

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = jnp.expand_dims(mask, 1) * jnp.expand_dims(mask, 2)
        dX = jnp.expand_dims(X, 1) - jnp.expand_dims(X, 2)
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, 3) + eps)
        #print(D[0])
        D_max = jnp.max(D, -1, keepdims=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = jax.lax.approx_min_k(D_adjust,
                                                  np.minimum(self.top_k, X.shape[1]),
                                                  reduction_dimension=-1)
        return D_neighbors, E_idx

    def _rbf(self, D):
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_mu = D_mu.reshape([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = jnp.expand_dims(D, -1)
        RBF = jnp.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = jnp.sqrt(jnp.sum((A[:,:,None,:] - B[:,None,:,:])**2, -1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def __call__(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0:
            use_key = hk.next_rng_key()
            #self.safe_key, use_key = self.safe_key.split()
            X = X + self.augment_eps * jax.random.normal(use_key, X.shape)

        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = jnp.cross(b, c)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = jnp.concatenate(tuple(RBF_all), axis=-1)

        offset = residue_idx[:,:,None] - residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = (chain_labels[:, :, None] - chain_labels[:,None,:])==0
        d_chains = jax.lax.convert_element_type(d_chains, jnp.int64)
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(jax.lax.convert_element_type(offset, jnp.int64), E_chains)
        E = jnp.concatenate((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx

Gelu = functools.partial(jax.nn.gelu, approximate=False)
class PositionWiseFeedForward(hk.Module):
    def __init__(self, num_hidden, num_ff, name=None):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = hk.Linear(num_ff, with_bias=True, name=name + '_W_in')
        self.W_out = hk.Linear(num_hidden, with_bias=True, name=name + '_W_out')
        self.act = Gelu
    def __call__(self, h_V):
        h = self.act(self.W_in(h_V), approximate=False)
        h = self.W_out(h)
        return h
class dropout_cust(hk.Module):
    def __init__(self, rate) -> None:
        super().__init__()
        self.rate = rate
        self.safe_key = SafeKey(hk.next_rng_key())

    def __call__(self, x):
        self.safe_key, use_key = self.safe_key.split()
        return hk.dropout(use_key.get(), self.rate, x)
class EncLayer(hk.Module):
    def __init__(self, num_hidden,
                 num_in, dropout=0.1,
                 num_heads=None, scale=30,
                 name=None):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.safe_key = SafeKey(hk.next_rng_key())

        self.dropout1 = dropout_cust(dropout)
        self.dropout2 = dropout_cust(dropout)
        self.dropout3 = dropout_cust(dropout)
        self.norm1 = hk.LayerNorm(-1, create_scale=True, create_offset=True,
                                  name=name + '_norm1')
        self.norm2 = hk.LayerNorm(-1, create_scale=True, create_offset=True,
                                  name=name + '_norm2')
        self.norm3 = hk.LayerNorm(-1, create_scale=True, create_offset=True,
                                  name=name + '_norm3')

        self.W1 = hk.Linear(num_hidden, with_bias=True, name=name + '_W1')
        self.W2 = hk.Linear(num_hidden, with_bias=True, name=name + '_W2')
        self.W3 = hk.Linear(num_hidden, with_bias=True, name=name + '_W3')
        self.W11 = hk.Linear(num_hidden, with_bias=True, name=name + '_W11')
        self.W12 = hk.Linear(num_hidden, with_bias=True, name=name + '_W12')
        self.W13 = hk.Linear(num_hidden, with_bias=True, name=name + '_W13')
        self.act = Gelu
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4,
                                             name=name + '_dense')

    def __call__(self, h_V, h_E, E_idx,
                mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        #h_V_expand = jnp.expand_dims(h_V, -2).tile([1, 1, h_EV.shape[-2], 1])
        h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2),[1, 1, h_EV.shape[-2], 1])
        h_EV = jnp.concatenate([h_V_expand, h_EV], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = jnp.expand_dims(mask_attend, -1)* h_message
        dh = jnp.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = jnp.expand_dims(mask_V, -1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        #h_V_expand = jnp.expand_dims(h_V, -2).tile([1, 1, h_EV.shape[-2], 1])
        h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2),[1, 1, h_EV.shape[-2], 1])
        h_EV = jnp.concatenate([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(hk.Module):
    """Decoder layer for autoregressive sequence generation."""
    def __init__(self, num_hidden, num_in,
                 dropout=0.1, num_heads=None,
                 scale=30, name=None):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = dropout_cust(dropout)
        self.dropout2 = dropout_cust(dropout)
        self.norm1 = hk.LayerNorm(-1, create_scale=True, create_offset=True,
                                  name=name + '_norm1')
        self.norm2 = hk.LayerNorm(-1, create_scale=True, create_offset=True,
                                  name=name + '_norm2')

        self.W1 = hk.Linear(num_hidden, with_bias=True, name=name + '_W1')
        self.W2 = hk.Linear(num_hidden, with_bias=True, name=name + '_W2')
        self.W3 = hk.Linear(num_hidden, with_bias=True, name=name + '_W3')
        self.act = Gelu
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4,
                                             name=name + '_dense')

    def __call__(self, h_V, h_E,
                 mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # Concatenate h_V_i to h_E_ij
        h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, 1, h_E.shape[-2], 1])
        h_EV = jnp.concatenate([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = jnp.expand_dims(mask_attend, -1) * h_message
        dh = jnp.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = jnp.expand_dims(mask_V, -1)
            h_V = mask_V * h_V
        return h_V


class EmbedToken(hk.Module):
    """Token embedding layer for sequence encoding."""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.w_init = hk.initializers.TruncatedNormal()

    @property
    def embeddings(self):
        return hk.get_parameter("W_s",
                                [self.vocab_size, self.embed_dim],
                                init=self.w_init)

    def __call__(self, arr):
        if jnp.issubdtype(arr.dtype, jnp.integer):
            one_hot = jax.nn.one_hot(arr, self.vocab_size)
        else:
            one_hot = arr
        return jnp.tensordot(one_hot, self.embeddings, 1)


class ENC:
      def __init__(self,  node_features,
                 edge_features, hidden_dim,
                 num_encoder_layers=1,
                  k_neighbors=64,
                 augment_eps=0.05, dropout=0.1):
        super(ENC, self).__init__()
        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        # Featurization layers
        self.features = ProteinFeatures(node_features,
                                        edge_features,
                                        top_k=k_neighbors,
                                        augment_eps=augment_eps)

        self.W_e = hk.Linear(hidden_dim, with_bias=True, name='W_e')
        # Encoder layers
        self.encoder_layers = [
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout, name='enc' + str(i))
            for i in range(num_encoder_layers)
        ]
      def __call__(self, X,
                 mask,
                 residue_idx, chain_encoding_all,
                 ):
        """ Graph-conditioned sequence model """
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = jnp.zeros((E.shape[0], E.shape[1], E.shape[-1]))
        h_E = self.W_e(E)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(jnp.expand_dims(mask, -1),  E_idx).squeeze(-1)
        mask_attend = jnp.expand_dims(mask, -1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        return h_V


class ENC_DEC:
    """Encoder-Decoder model for protein sequence design.

    This class combines the encoder from SoftAlign with the decoder from
    ColabDesign's ProteinMPNN. It is backwards compatible with ENC - when
    num_decoder_layers=0, it behaves identically to ENC.

    The decoder enables autoregressive sequence generation given structure.
    """
    def __init__(self, node_features, edge_features, hidden_dim,
                 num_encoder_layers=1, num_decoder_layers=0,
                 k_neighbors=64, augment_eps=0.05, dropout=0.1,
                 vocab=21, num_letters=21):
        super(ENC_DEC, self).__init__()
        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        self.vocab = vocab

        # Featurization layers
        self.features = ProteinFeatures(node_features,
                                        edge_features,
                                        top_k=k_neighbors,
                                        augment_eps=augment_eps)

        self.W_e = hk.Linear(hidden_dim, with_bias=True, name='W_e')

        # Encoder layers
        self.encoder_layers = [
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout, name='enc' + str(i))
            for i in range(num_encoder_layers)
        ]

        # Decoder layers (optional, for sequence design)
        # Decoder input is [h_VS_expand, h_ES] where h_ES = [h_E, h_S_neighbors, h_V_neighbors]
        # So input dim = hidden_dim + hidden_dim*3 = hidden_dim*4 = 512 for hidden_dim=128
        if num_decoder_layers > 0:
            self.W_s = EmbedToken(vocab_size=vocab, embed_dim=hidden_dim)
            self.decoder_layers = [
                DecLayer(hidden_dim, hidden_dim*4, dropout=dropout, name='dec' + str(i))
                for i in range(num_decoder_layers)
            ]
            self.W_out = hk.Linear(num_letters, with_bias=True, name='W_out')
        else:
            self.W_s = None
            self.decoder_layers = []
            self.W_out = None

    def encode(self, X, mask, residue_idx, chain_encoding_all):
        """Run encoder only - same as ENC.__call__"""
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = jnp.zeros((E.shape[0], E.shape[1], E.shape[-1]))
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(jnp.expand_dims(mask, -1), E_idx).squeeze(-1)
        mask_attend = jnp.expand_dims(mask, -1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        return h_V, h_E, E_idx

    def decode(self, h_V, h_E, E_idx, S, mask):
        """Run decoder on encoded features.

        Args:
            h_V: Node embeddings from encoder [B, L, hidden_dim]
            h_E: Edge embeddings from encoder [B, L, K, hidden_dim]
            E_idx: Edge indices [B, L, K]
            S: Sequence indices [B, L] (integer amino acid indices)
            mask: Mask for valid positions [B, L]

        Returns:
            logits: Amino acid logits [B, L, vocab]
        """
        if self.num_decoder_layers == 0:
            raise ValueError("No decoder layers configured. Set num_decoder_layers > 0.")

        # Embed sequence
        h_S = self.W_s(S)

        # Concatenate sequence embedding to h_V
        h_VS = h_V + h_S

        # Prepare edge features for decoder
        # h_ES = [h_E, gathered_h_S, gathered_h_V] to get hidden_dim*3 = 384
        # Then in decoder: h_EV = [h_VS_expand, h_ES] = hidden_dim*4 = 512
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)  # [h_E, h_S_neighbors] = 256
        h_EV_neighbors = gather_nodes(h_V, E_idx)    # h_V at neighbor positions = 128
        h_ES = jnp.concatenate([h_ES, h_EV_neighbors], -1)  # 256 + 128 = 384

        # Decoder mask
        mask_attend = gather_nodes(jnp.expand_dims(mask, -1), E_idx).squeeze(-1)
        mask_attend = jnp.expand_dims(mask, -1) * mask_attend

        # Run decoder layers
        for layer in self.decoder_layers:
            h_VS = layer(h_VS, h_ES, mask, mask_attend)

        # Output logits
        logits = self.W_out(h_VS)
        return logits

    def __call__(self, X, mask, residue_idx, chain_encoding_all, S=None):
        """Forward pass.

        If S is None, returns only encoder output (backwards compatible with ENC).
        If S is provided and decoder is configured, returns logits.
        """
        h_V, h_E, E_idx = self.encode(X, mask, residue_idx, chain_encoding_all)

        if S is None or self.num_decoder_layers == 0:
            # Backwards compatible: return encoder output only
            return h_V
        else:
            # Full encoder-decoder: return logits
            logits = self.decode(h_V, h_E, E_idx, S, mask)
            return logits


import os as _os

# Default path to ProteinMPNN weights (relative to this module)
_MODELS_DIR = _os.path.join(_os.path.dirname(__file__), 'models')
DEFAULT_MPNN_WEIGHTS = 'v_48_020'  # Use variant name, resolution happens in load function

# Available ProteinMPNN weight variants (base names without extension)
MPNN_WEIGHT_VARIANTS = ['v_48_002', 'v_48_010', 'v_48_020', 'v_48_030']


def load_colabdesign_weights(init_params, weights_path=None):
    """Load ColabDesign ProteinMPNN weights into SoftAlign ENC_DEC model.

    This function maps parameter names from ColabDesign's naming convention
    (with 'protein_mpnn/~/' prefix) to SoftAlign's naming convention.

    Args:
        init_params: Initialized parameters from hk.transform(...).init()
        weights_path: Path to weights file, or name of variant
                      (e.g., 'v_48_020'). If None, uses default weights.
                      Available variants: v_48_002, v_48_010, v_48_020, v_48_030

    Returns:
        Dictionary of parameters that can be passed to transformed.apply()

    Example:
        import haiku as hk
        from softalign import MPNN

        def forward_fn(X, mask, residue_idx, chain_idx, S):
            model = MPNN.ENC_DEC(
                node_features=128, edge_features=128, hidden_dim=128,
                num_encoder_layers=3, num_decoder_layers=3,
                k_neighbors=48, augment_eps=0.0, dropout=0.0
            )
            return model(X, mask, residue_idx, chain_idx, S)

        transformed = hk.transform(forward_fn)
        init_params = transformed.init(rng, X, mask, residue_idx, chain_idx, S)

        # Use default weights
        params = MPNN.load_colabdesign_weights(init_params)

        # Or specify a variant
        params = MPNN.load_colabdesign_weights(init_params, 'v_48_010')

        output = transformed.apply(params, rng, X, mask, residue_idx, chain_idx, S)
    """
    # Resolve weights path
    if weights_path is None:
        weights_path = DEFAULT_MPNN_WEIGHTS

    # If it's a variant name, resolve to full path (without extension)
    if weights_path in MPNN_WEIGHT_VARIANTS:
        base_path = _os.path.join(_MODELS_DIR, weights_path)
    else:
        # Remove extension if present for consistent handling
        base_path = weights_path
        if base_path.endswith('.npz'):
            base_path = base_path[:-4]

    npz_path = base_path + '.npz'
    data = dict(np.load(npz_path, allow_pickle=False))
    # Unflatten the dictionary structure
    checkpoint = utils.unflatten_dict(data)
    loaded_params = checkpoint.get('model_state_dict', checkpoint)
    # Convert numpy arrays to JAX arrays
    loaded_params = utils.convert_numpy_to_jax(loaded_params)

    new_params = {}
    cd_prefix = 'protein_mpnn/~/'

    for our_key in init_params.keys():
        cd_key = cd_prefix + our_key
        if cd_key in loaded_params:
            new_params[our_key] = loaded_params[cd_key]
        else:
            raise KeyError(f"No matching weights found for '{our_key}' "
                          f"(looked for '{cd_key}')")

    return new_params


def create_enc_dec_model(hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                         k_neighbors=48, augment_eps=0.0, dropout=0.0,
                         vocab=21, num_letters=21):
    """Create an ENC_DEC model with configuration matching ColabDesign ProteinMPNN.

    This is a convenience function that creates an ENC_DEC model with the same
    default configuration as ColabDesign's ProteinMPNN, making it easy to load
    pre-trained weights.

    Args:
        hidden_dim: Dimension of hidden layers (default 128 to match ProteinMPNN)
        num_encoder_layers: Number of encoder layers (default 3)
        num_decoder_layers: Number of decoder layers (default 3)
        k_neighbors: Number of nearest neighbors in graph (default 48)
        augment_eps: Backbone noise augmentation (default 0.0)
        dropout: Dropout rate (default 0.0)
        vocab: Vocabulary size for input sequences (default 21)
        num_letters: Output vocabulary size (default 21)

    Returns:
        ENC_DEC model instance
    """
    return ENC_DEC(
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        k_neighbors=k_neighbors,
        augment_eps=augment_eps,
        dropout=dropout,
        vocab=vocab,
        num_letters=num_letters
    )
