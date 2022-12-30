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
    ['void(float64[:, :], float64[:], float64[:, :])'],
    '(n, m),(l)->(n, m)',
    target='parallel',
    nopython=True
)
def normalize_sym_V(sym_V, sym_V_eq, result):
    for i in range(sym_V.shape[0]):
        if sym_V[i, 0] != sym_V[i, 1]:
            result[i, 0] = sym_V[i, 0]
            result[i, 1] = sym_V[i, 1]
            result[i, 2] = 2 * sym_V[i, 2] / (sym_V_eq[int(sym_V[i, 0])] + sym_V_eq[int(sym_V[i, 1])])
            result[i, 3] = sym_V[i, 3]
        else:
            result[i, 0] = sym_V[i, 0]
            result[i, 1] = sym_V[i, 1]
            result[i, 2] = sym_V[i, 2]
            result[i, 3] = sym_V[i, 3]


print('Reading sym_V matrix from .hdf5 file...')
f = h5py.File('/home/mashkova/ortologs/7_species_res/sym_V.hdf5')
d = f['/x'] 
sym_V_da = da.from_array(d, chunks=CHUNK_NUM)
sym_V_df = dd.from_dask_array(sym_V_da, columns=['qseqid', 'sseqid', 'score', 'evalue'])

print('Reading protein names from .npz file...')
protein_names = np.load('/home/mashkova/ortologs/7_species_res/protein_names_sym_V.npz', allow_pickle=True)['x']

print('Decoding protein names...')
sym_V_df['qseqid'] = sym_V_df['qseqid'].apply(lambda x: protein_names[int(x)], meta=('qseqid', 'object'))
sym_V_df['sseqid'] = sym_V_df['sseqid'].apply(lambda x: protein_names[int(x)], meta=('sseqid', 'object'))

print('Sorting V matrix by qseqid and sseqid...')
sym_V_df_qseqid = sym_V_df.compute().sort_values(['qseqid', 'sseqid'])

print('Encoding protein names...')
le = LabelEncoder()
le.fit(np.unique(sym_V_df_qseqid[['qseqid', 'sseqid']].values.flatten()))
sym_V_df_qseqid[['qseqid', 'sseqid']] = sym_V_df_qseqid[['qseqid', 'sseqid']].apply(le.fit_transform)

print('Saving protein names to .npz file...')
np.savez('/home/mashkova/ortologs/7_species_res/protein_names_norm_sym_V', x=protein_names)

print('Extracting values for normalization...')
sym_V_eq_df = sym_V_df_qseqid[sym_V_df_qseqid.qseqid == sym_V_df_qseqid.sseqid].score

print('Preparing sym_V for normalization...')
sym_V = da.from_array(sym_V_df_qseqid.values, chunks=CHUNK_NUM).astype('float64')
sym_V_eq = da.from_array(sym_V_eq_df.values, chunks=CHUNK_NUM).astype('float64')

print('Computing normaized sym_V matrix...')
result = delayed(normalize_sym_V)(sym_V, sym_V_eq)
print('Done!')

print('Saving normalized sym_V matrix to .hdf5 file...')
da_array = da.from_delayed(result, sym_V.shape, sym_V.dtype)
da_array = da_array.rechunk(chunks=CHUNK_NUM)
da.to_hdf5('/home/mashkova/ortologs/7_species_res/norm_sym_V.hdf5', '/x', da_array)
print('Done!')
