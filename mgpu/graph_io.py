import pickle
import numpy as np
import scipy.sparse as sp

# return a scipy.sparse.csr_matrix
def load_graph(filename):
    with open(filename + '-indptr.pkl', 'rb') as f:
        indptr = pickle.load(f)
    with open(filename + '-indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    N = indptr.shape[0] - 1
    M = indices.shape[0]
    data = np.ones((M,), dtype=np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(N, N), dtype=np.float32)

# assume mg_int is int32_t
# g is in scipy.sparse.csr_matrix
def save_graph(filename, g):
    with open(filename + '-indptr.pkl', 'wb') as f:
        pickle.dump(g.indptr, f) 
    with open(filename + '-indices.pkl', 'wb') as f:
        pickle.dump(g.indices, f) 
