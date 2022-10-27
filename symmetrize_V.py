import dask.dataframe as dd
import dask.array as da
from dask import delayed, compute
import numpy as np
from numba import guvectorize
import h5py
from sklearn.preprocessing import LabelEncoder
from dask.diagnostics import ProgressBar

CHUNK_NUM = 1000000

ProgressBar().register()

@guvectorize(
    ['void(float64[:, :], float64[:, :], float64[:, :])'],
    '(n, m),(n, m)->(n,m)',
    target='parallel',
    nopython=True
)
def symmetrize_V(V, sym_V, result):
    i = 0 
    j = 0
    while i < sym_V.shape[0]:
        if sym_V[i, 0] == V[j, 0] == sym_V[i, 1] == V[j, 1]:
            result[i, 0] = sym_V[i, 0]
            result[i, 1] = sym_V[i, 1]
            result[i, 2] = sym_V[i, 2]
            result[i, 3] = sym_V[i, 3]
            i += 1
            j += 1
        elif sym_V[i, 0] == V[j, 1] and sym_V[i, 1] == V[j, 0]:
            result[i, 0] = sym_V[i, 0]
            result[i, 1] = sym_V[i, 1]
            result[i, 2] = min(sym_V[i, 2], V[j, 2])
            result[i, 3] = max(sym_V[i, 3], V[j, 3])
            i += 1
            j += 1
        else:
            qseqid_1 = sym_V[i, 1]
            sseqid_1 = sym_V[i, 0]
            qseqid_2 = V[j, 1]
            sseqid_2 = V[j, 0]
            if (V[j-1, 1] < sseqid_1 and sseqid_1 < V[j, 1]) or (V[j-1, 1] < sseqid_1 and sseqid_1 == V[j, 1] and qseqid_1 <= V[j, 0]) or (V[j-1, 1] == sseqid_1 and V[j-1, 0] <= qseqid_1 and sseqid_1 < V[j, 1]) or (V[j-1, 1] == sseqid_1 and V[j-1, 0] <= qseqid_1 and sseqid_1 == V[j, 1] and qseqid_1 <= V[j, 0]):
                result[i, 0] = sym_V[i, 0]
                result[i, 1] = sym_V[i, 1]
                result[i, 2] = sym_V[i, 2]
                result[i, 3] = sym_V[i, 3]
                i += 1
            elif (sym_V[i-1, 0] < qseqid_2 and qseqid_2 < sym_V[i, 0]) or (sym_V[i-1, 0] < qseqid_2 and qseqid_2 == sym_V[i, 0] and sseqid_2 <= sym_V[i, 1]) or (sym_V[i-1, 0] == qseqid_2 and sym_V[i-1, 1] <= sseqid_2 and qseqid_2 < sym_V[i, 0]) or (sym_V[i-1, 0] == qseqid_2 and sym_V[i-1, 1] <= sseqid_2 and qseqid_2 == sym_V[i, 0] and sseqid_2 <= sym_V[i, 1]):
                j += 1               


print('Reading V matrix from .hdf5 file...')
f = h5py.File('/home/mashkova/ortologs/7_species_res/V.hdf5')
d = f['/x'] 
V_da = da.from_array(d, chunks=CHUNK_NUM)
V_df = dd.from_dask_array(V_da, columns=['qseqid', 'sseqid', 'score', 'evalue'])

print('Reading protein names from .npz file...')
protein_names = np.load('/home/mashkova/ortologs/7_species_res/protein_names_V.npz', allow_pickle=True)['x']

print('Decoding protein names...')
V_df['qseqid'] = V_df['qseqid'].apply(lambda x: protein_names[int(x)], meta=('qseqid', 'object'))
V_df['sseqid'] = V_df['sseqid'].apply(lambda x: protein_names[int(x)], meta=('sseqid', 'object'))

print('Sorting V matrix by qseqid and sseqid...')
V_df_sorted = V_df.compute().sort_values(['qseqid', 'sseqid'])
V_df_sorted_2 = V_df.compute().sort_values(['sseqid', 'qseqid'])

print('Encoding protein names...')
le = LabelEncoder()
le.fit(np.unique(V_df_sorted[['qseqid', 'sseqid']].values.flatten()))
V_df_sorted[['qseqid', 'sseqid']] = V_df_sorted[['qseqid', 'sseqid']].apply(le.fit_transform)
V_df_sorted_2[['sseqid', 'qseqid']] = V_df_sorted_2[['sseqid', 'qseqid']].apply(le.fit_transform)

print('Saving protein names to .npz file...')
np.savez('/home/mashkova/ortologs/7_species_res/protein_names_sym_V', x=protein_names)

print('Prepairing data for symmetrization...')
V = da.from_array(V_df_sorted_2.values, chunks=CHUNK_NUM).astype('float64')
sym_V = da.from_array(V_df_sorted.values, chunks=CHUNK_NUM).astype('float64')

print('Computing symmetrized V matrix...')
result = delayed(symmetrize_V)(V, sym_V)
print('Done!')

print('Saving symmetrized V matrix to .hdf5 file...')
da_array = da.from_delayed(result, V.shape, V.dtype)
da.to_hdf5('/home/mashkova/ortologs/7_species_res/sym_V.hdf5', '/x', da_array)
print('Done!')