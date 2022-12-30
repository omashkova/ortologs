import dask.dataframe as dd
import dask.array as da
from dask import compute
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import LabelEncoder
from dask.diagnostics import ProgressBar

R = 'Xenopus_tropicalis'
u = 3

CHUNK_NUM = 1000000

ProgressBar().register()

print('Reading W matrix from .hdf5 file...')
f = h5py.File('/home/mashkova/ortologs/18_species_res/W.hdf5')
d = f['/x'] 
W_da = da.from_array(d, chunks=CHUNK_NUM)
W_df = dd.from_dask_array(W_da, columns=['qseqid_gene', 'sseqid_gene', 'score', 'evalue', 'qseqid_species', 'sseqid_species'])

print('Reading gene names from .npz file...')
gene_names = np.load('/home/mashkova/ortologs/18_species_res/gene_names_W.npz', allow_pickle=True)['x']

print('Decoding gene names...')
W_df['qseqid_gene'] = W_df['qseqid_gene'].apply(lambda x: gene_names[int(x)], meta=('qseqid_gene', 'object'))
W_df['sseqid_gene'] = W_df['sseqid_gene'].apply(lambda x: gene_names[int(x)], meta=('sseqid_gene', 'object'))

print('Reading species names from .npz file...')
species_names = np.load('/home/mashkova/ortologs/18_species_res/species_names_W.npz', allow_pickle=True)['x']

print('Decoding species names...')
W_df['qseqid_species'] = W_df['qseqid_species'].apply(lambda x: species_names[int(x)], meta=('qseqid_species', 'object'))
W_df['sseqid_species'] = W_df['sseqid_species'].apply(lambda x: species_names[int(x)], meta=('sseqid_species', 'object'))

print('Prepare homology finder...')
finder = W_df[(W_df['qseqid_species'] == R) & (W_df['sseqid_species'] != R)]
sorted_finder = finder.compute().sort_values(by=['qseqid_gene', 'score'], ascending=[True, False])

print('Get alpha homologs...')
df = sorted_finder.loc[lambda x: x.groupby('qseqid_gene', group_keys=False)['score'].nlargest(u).index]

print('Encoding gene names...')
le = LabelEncoder()
le.fit(np.unique(df[['qseqid_gene', 'sseqid_gene']].values.flatten()))
df[['qseqid_gene', 'sseqid_gene']] = df[['qseqid_gene', 'sseqid_gene']].apply(le.transform)

print('Saving gene names to .npz file...')
np.savez('/home/mashkova/ortologs/18_species_res/gene_names_alpha_homology', x=le.classes_)

print("Encoding species names...")
le = LabelEncoder()
le.fit(np.unique(df[['qseqid_species', 'sseqid_species']].values.flatten()))
df[['qseqid_species', 'sseqid_species']] = df[['qseqid_species', 'sseqid_species']].apply(le.transform)

print('Saving species names to .npz file...')
np.savez('/home/mashkova/ortologs/18_species_res/species_names_alpha_homology', x=le.classes_)

print('Saving alpha homology matrix to .hdf5 file...')
alpha_homology = df.values.astype('float64')
dasked_alpha_homology = da.from_array(alpha_homology, chunks=CHUNK_NUM)
dasked_alpha_homology.to_hdf5('/home/mashkova/ortologs/18_species_res/alpha_homology.hdf5', '/x') 
print('Done!')