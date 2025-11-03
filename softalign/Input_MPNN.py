import jax
import jax.numpy as jnp
import numpy as np

# From colabfold
MODRES = {'MSE':'MET','MLY':'LYS','FME':'MET','HYP':'PRO',
          'TPO':'THR','CSO':'CYS','SEP':'SER','M3L':'LYS',
          'HSK':'HIS','SAC':'SER','PCA':'GLU','DAL':'ALA',
          'CME':'CYS','CSD':'CYS','OCS':'CYS','DPR':'PRO',
          'B3K':'LYS','ALY':'LYS','YCM':'CYS','MLZ':'LYS',
          '4BF':'TYR','KCX':'LYS','B3E':'GLU','B3D':'ASP',
          'HZP':'PRO','CSX':'CYS','BAL':'ALA','HIC':'HIS',
          'DBZ':'ALA','DCY':'CYS','DVA':'VAL','NLE':'LEU',
          'SMC':'CYS','AGM':'ARG','B3A':'ALA','DAS':'ASP',
          'DLY':'LYS','DSN':'SER','DTH':'THR','GL3':'GLY',
          'HY3':'PRO','LLP':'LYS','MGN':'GLN','MHS':'HIS',
          'TRQ':'TRP','B3Y':'TYR','PHI':'PHE','PTR':'TYR',
          'TYS':'TYR','IAS':'ASP','GPL':'LYS','KYN':'TRP',
          'CSD':'CYS','SEC':'CYS'}

def _np_norm(x, axis=-1, keepdims=True, eps=1e-8, use_jax=True):
  '''compute norm of vector'''
  _np = jnp if use_jax else np
  return _np.sqrt(_np.square(x).sum(axis,keepdims=keepdims) + 1e-8)
def _np_extend(a,b,c, L,A,D, use_jax=True):
  '''
  given coordinates a-b-c,
  c-d (L)ength, b-c-d (A)ngle, and a-b-c-d (D)ihedral
  return 4th coordinate d
  '''
  _np = jnp if use_jax else np
  normalize = lambda x: x/_np_norm(x, use_jax=use_jax)
  bc = normalize(b-c)
  n = normalize(_np.cross(b-a, bc))
  return c + sum([L * _np.cos(A) * bc,
                  L * _np.sin(A) * _np.cos(D) * _np.cross(n, bc),
                  L * _np.sin(A) * _np.sin(D) * -n])

def _np_get_cb(N,CA,C, use_jax=True):
  '''compute CB placement from N, CA, C'''
  return _np_extend(C, N, CA, 1.522, 1.927, -2.143, use_jax=use_jax)
def get_pdb(x, chains=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''
  restype_3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
                  'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
                  'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}

  atoms = ['N','CA','C','CB']

  coords, seq, ids = {},[],[]
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    # if line[:4] == "ATOM" or line[:6] == "HETATM":
    if line[:6] == "HETATM":
      ch = line[21:22]
      if chains is None or ch in chains:
        resi = line[17:20].strip()
        if resi in MODRES:
          new_res = MODRES[resi]
          # turn HETATM into ATOM and replace modified residue name with canonical one
          line = "ATOM  " + line[6:]
          line = line[:17] + new_res.ljust(3) + line[20:]
    if line[:4] == "ATOM":
      ch = line[21:22]
      if chains is None or ch in chains:
        resi, resn = line[17:17+3], line[22:22+5].strip()
        # resn = int(resn[:-1] if resn[-1].isalpha() else resn)
        
        id = f"{resn}"
        if id not in ids:
          coords[id] = {}
          seq.append(restype_3to1.get(resi,"X"))
          ids.append(id)          

        atom = line[12:12+4].strip()
        if atom in atoms and atom not in coords[id]:
          coords[id][atom] = np.array([float(line[i:(i+8)]) for i in [30,38,46]])

    if line[:6] == "ENDMDL":
      break
  
  # get xyz
  xyz = []
  for id in ids:
    if "CA" in coords[id]: # only include residues that contain CA
      xyz.append(np.full((len(atoms),3),np.nan))
      for a,atom in enumerate(atoms):
        if atom in coords[id]:
          xyz[-1][a] = coords[id][atom]
  xyz = np.array(xyz)

  # add CB coordinate
  xyz[:,3] = _np_get_cb(xyz[:,0], xyz[:,1], xyz[:,2], use_jax=False)

  return {"xyz":xyz,
          "seq":np.array(seq),
          "ids":np.array(ids)}

def get_inputs_mpnn(pdb_path,chain = None):

    data = get_pdb(pdb_path, chain)
    coords = data["xyz"]
    ids = data["ids"]
    cond = ~np.isnan(coords).any(axis=(1,2))
    mask_ = np.ones(coords.shape[0])[cond]
    chain_ = np.ones(coords.shape[0])[cond]
    res = np.arange(coords.shape[0])[cond]
    coords = coords[cond,:]
    return coords[None,:],mask_[None,:],chain_[None,:],res[None,:], ids
