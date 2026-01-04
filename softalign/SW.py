# Code from the Smooth-Smith-Waterman paper
# Written by Sergey Ovchinnikov and Sam Petti
# Spring 2021
import jax
import jax.numpy as jnp

def sw(unroll=2, batch=True, NINF=-1e30):
  '''smith-waterman (local alignment) with gap parameter'''

  # rotate matrix for striped dynamic-programming
  def rotate(x):
    # solution from jake vanderplas (thanks!)
    a,b = x.shape
    ar,br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
    i,j = (br-ar)+(a-1),(ar+br)//2
    n,m = (a+b-1),(a+b)//2
    output = {"x":jnp.full([n,m],NINF).at[i,j].set(x), "o":(jnp.arange(n)+a%2)%2}
    return output, (jnp.full(m, NINF), jnp.full(m, NINF)), (i,j)

  # compute scoring (hij) matrix
  def sco(x, lengths, gap=0, temp=1.0):

    def _soft_maximum(x, axis=None, mask=None):
      def _logsumexp(y):
        y = jnp.maximum(y,NINF)
        if mask is None: return jax.nn.logsumexp(y, axis=axis)
        else: return y.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(y - y.max(axis, keepdims=True)), axis=axis))
      return temp*_logsumexp(x/temp)

    def _cond(cond, true, false): return cond*true + (1-cond)*false
    def _pad(x,shape): return jnp.pad(x,shape,constant_values=(NINF,NINF))

    def _step(prev, sm):
      h2,h1 = prev   # previous two rows of scoring (hij) mtx
      h1_T = _cond(sm["o"],_pad(h1[:-1],[1,0]),_pad(h1[1:],[0,1]))

      # directions
      Align = h2 + sm["x"]
      Turn_0 = h1 + gap
      Turn_1 = h1_T + gap
      Sky = sm["x"]

      h0 = jnp.stack([Align, Turn_0, Turn_1, Sky], -1)
      h0 = _soft_maximum(h0, -1)
      return (h1,h0),h0

    # mask
    a,b = x.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a) < real_a)[:,None] * (jnp.arange(b) < real_b)[None,:]
    x = x + NINF * (1 - mask)

    sm, prev, idx = rotate(x[:-1,:-1])
    hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]
    return _soft_maximum(hij + x[1:,1:], mask=mask[1:,1:])

  # traceback (aka backprop) to get alignment
  traceback = jax.value_and_grad(sco)

  # add batch dimension
  if batch: return jax.vmap(traceback,(0,0,None,None))
  else: return traceback


def sw_affine(restrict_turns=True,
             penalize_turns=True,
             batch=True, unroll=2, NINF=-1e30):
  """smith-waterman (local alignment) with affine gap

  Args:
    restrict_turns: If True, restrict state transitions
    penalize_turns: If True, use different penalties for gap open/extend
    batch: If True, add batch dimension via vmap
    unroll: Loop unrolling factor for jax.lax.scan
    NINF: Negative infinity value for masking

  Returns:
    Function that takes (x, lengths, gap, open, temp, gap_matrix, open_matrix, penalize_start_gap)
    where gap_matrix and open_matrix are optional position-dependent penalties,
    and penalize_start_gap enables N-terminal start gap penalties.
  """
  # rotate matrix for vectorized dynamic-programming

  def rotate(x, fill_value=NINF):
    # solution from jake vanderplas (thanks!)
    a,b = x.shape
    ar,br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
    i,j = (br-ar)+(a-1),(ar+br)//2
    n,m = (a+b-1),(a+b)//2
    output = {"x":jnp.full([n,m],fill_value).at[i,j].set(x), "o":(jnp.arange(n)+a%2)%2}
    return output, (jnp.full((m,3), NINF), jnp.full((m,3), NINF)), (i,j)

  def rotate_gap_matrix(x):
    """Rotate a gap penalty matrix for striped DP, using 0 as fill value."""
    a,b = x.shape
    ar,br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
    i,j = (br-ar)+(a-1),(ar+br)//2
    n,m = (a+b-1),(a+b)//2
    # Use 0.0 as fill value for gap matrices (no penalty outside valid region)
    return jnp.full([n,m], 0.0).at[i,j].set(x)

  def rotate_penalty(penalty_1d, a, b):
    """Rotate a 1D penalty array to match the striped DP format.

    Args:
      penalty_1d: 1D array of shape (b,) with penalty for each reference column
      a, b: Original matrix dimensions

    Returns:
      2D rotated penalty array of shape (n, m) matching the DP format
    """
    ar,br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
    i,j = (br-ar)+(a-1),(ar+br)//2
    n,m = (a+b-1),(a+b)//2
    # Create penalty matrix by broadcasting the 1D penalty across rows
    penalty_2d = jnp.broadcast_to(penalty_1d, (a, b))
    # Rotate to match DP format, using 0.0 as fill value
    return jnp.full([n,m], 0.0).at[i,j].set(penalty_2d)

  # fill the scoring matrix
  def sco(x, lengths, gap=0.0, open=0.0, temp=1.0,
          gap_matrix=None, open_matrix=None, penalize_start_gap=False):
    """
    Args:
      x: Similarity matrix (query_len, target_len)
      lengths: Tuple of (real_query_len, real_target_len)
      gap: Scalar gap extension penalty (used if gap_matrix is None)
      open: Scalar gap open penalty (used if open_matrix is None)
      temp: Temperature for soft maximum
      gap_matrix: Optional position-dependent gap extension penalties (query_len, target_len)
      open_matrix: Optional position-dependent gap open penalties (query_len, target_len)
      penalize_start_gap: If True, penalize alignments starting after
        reference position 1. Penalty = open + gap * (j - 1) for j >= 1.
    """
    # Check if we're using position-dependent gap penalties
    use_matrix_gaps = gap_matrix is not None and open_matrix is not None

    def _soft_maximum(x, axis=None, mask=None):
      def _logsumexp(y):
        y = jnp.maximum(y,NINF)
        if mask is None: return jax.nn.logsumexp(y, axis=axis)
        else: return y.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(y - y.max(axis, keepdims=True)), axis=axis))
      return temp*_logsumexp(x/temp)

    def _cond(cond, true, false): return cond*true + (1-cond)*false
    def _pad(x,shape): return jnp.pad(x,shape,constant_values=(NINF,NINF))

    def _step_scalar(prev, sm):
      """Original step using scalar gap penalties."""
      h2,h1 = prev   # previous two rows of scoring (hij) mtxs

      Align = jnp.pad(h2,[[0,0],[0,1]]) + sm["x"][:,None]
      Right = _cond(sm["o"], _pad(h1[:-1],([1,0],[0,0])),h1)
      Down  = _cond(sm["o"], h1,_pad(h1[1:],([0,1],[0,0])))

      # add gap penalty
      if penalize_turns:
        Right += jnp.stack([open,gap,open])
        Down += jnp.stack([open,open,gap])
      else:
        gap_pen = jnp.stack([open,gap,gap])
        Right += gap_pen
        Down += gap_pen

      if restrict_turns: Right = Right[:,:2]

      h0_Align = _soft_maximum(Align,-1)
      h0_Right = _soft_maximum(Right,-1)
      h0_Down = _soft_maximum(Down,-1)
      h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
      return (h1,h0),h0

    def _step_matrix(prev, sm):
      """Step using position-dependent gap penalties from matrices."""
      h2,h1 = prev   # previous two rows of scoring (hij) mtxs

      Align = jnp.pad(h2,[[0,0],[0,1]]) + sm["x"][:,None]
      Right = _cond(sm["o"], _pad(h1[:-1],([1,0],[0,0])),h1)
      Down  = _cond(sm["o"], h1,_pad(h1[1:],([0,1],[0,0])))

      # Get position-dependent gap penalties from rotated matrices
      gap_vals = sm["gap"]    # shape: (m,)
      open_vals = sm["open"]  # shape: (m,)

      # add gap penalty (position-dependent)
      if penalize_turns:
        # Right: [open, gap, open] per position
        Right += jnp.stack([open_vals, gap_vals, open_vals], axis=-1)
        # Down: [open, open, gap] per position
        Down += jnp.stack([open_vals, open_vals, gap_vals], axis=-1)
      else:
        # [open, gap, gap] per position
        gap_pen = jnp.stack([open_vals, gap_vals, gap_vals], axis=-1)
        Right += gap_pen
        Down += gap_pen

      if restrict_turns: Right = Right[:,:2]

      h0_Align = _soft_maximum(Align,-1)
      h0_Right = _soft_maximum(Right,-1)
      h0_Down = _soft_maximum(Down,-1)
      h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
      return (h1,h0),h0

    def _step_with_sky(prev, sm):
      """Step function with Sky term for local alignment start with penalty."""
      h2,h1 = prev   # previous two rows of scoring (hij) mtxs

      # Pad with NINF (not 0) to prevent spurious paths through padded positions
      Align = jnp.pad(h2,[[0,0],[0,1]],constant_values=(NINF,NINF)) + sm["x"][:,None]
      Right = _cond(sm["o"], _pad(h1[:-1],([1,0],[0,0])),h1)
      Down  = _cond(sm["o"], h1,_pad(h1[1:],([0,1],[0,0])))

      # Sky term: start fresh at this position with start penalty
      # sm["start_pen"] is the rotated start penalty (negative for later columns)
      Sky = sm["x"] + sm["start_pen"]

      # add gap penalty
      if penalize_turns:
        Right += jnp.stack([open,gap,open])
        Down += jnp.stack([open,open,gap])
      else:
        gap_pen = jnp.stack([open,gap,gap])
        Right += gap_pen
        Down += gap_pen

      if restrict_turns: Right = Right[:,:2]

      # Include Sky in Align computation (it's another source for alignment state)
      # Sky contributes to starting a new alignment at this position
      Align_with_sky = jnp.concatenate([Align, Sky[:, None]], axis=-1)
      h0_Align = _soft_maximum(Align_with_sky, -1)
      h0_Right = _soft_maximum(Right,-1)
      h0_Down = _soft_maximum(Down,-1)
      h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
      return (h1,h0),h0

    # mask
    a,b = x.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a) < real_a)[:,None] * (jnp.arange(b) < real_b)[None,:]
    x = x + NINF * (1 - mask)

    sm, prev, idx = rotate(x[:-1,:-1])

    if use_matrix_gaps:
      # Rotate gap matrices the same way as similarity matrix
      rotated_gap = rotate_gap_matrix(gap_matrix[:-1,:-1])
      rotated_open = rotate_gap_matrix(open_matrix[:-1,:-1])
      sm["gap"] = rotated_gap
      sm["open"] = rotated_open
      hij = jax.lax.scan(_step_matrix, prev, sm, unroll=unroll)[-1][idx]
    else:
      # Pre-compute start penalty (cheap O(b) operation, needed for sky branch)
      # j=0: 0 (no penalty, column 0 is position 1)
      # j=1: open (one gap at position 1, starting at position 2)
      # j>=2: open + gap * (j - 1) (gap opening + extensions)
      col_indices = jnp.arange(b - 1)  # -1 because we use x[:-1,:-1]
      start_penalty = jnp.where(
          col_indices == 0,
          0.0,
          open + gap * (col_indices - 1)
      )
      rotated_penalty = rotate_penalty(start_penalty, a - 1, b - 1)

      # Use jax.lax.cond to avoid Python conditional on traced boolean
      # This allows penalize_start_gap to be a traced value in JIT
      def _run_with_sky():
        sm_sky = {"x": sm["x"], "o": sm["o"], "start_pen": rotated_penalty}
        return jax.lax.scan(_step_with_sky, prev, sm_sky, unroll=unroll)[-1][idx]

      def _run_scalar():
        return jax.lax.scan(_step_scalar, prev, sm, unroll=unroll)[-1][idx]

      hij = jax.lax.cond(penalize_start_gap, _run_with_sky, _run_scalar)

    # sink
    return _soft_maximum(hij + x[1:,1:,None], mask=mask[1:,1:,None])

  # traceback to get alignment (aka. get marginals)
  traceback = jax.value_and_grad(sco)

  # add batch dimension
  # Note: gap_matrix and open_matrix are batched along axis 0 if provided
  # penalize_start_gap is a scalar (not batched)
  if batch:
    return jax.vmap(traceback, (0, 0, None, None, None, 0, 0, None))
  else:
    return traceback
