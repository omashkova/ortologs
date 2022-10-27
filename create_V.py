import dask.dataframe as dd
import dask.array as da
import dask.bag as db
from dask import compute, delayed
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from dask.diagnostics import ProgressBar


ORGANISM_NAMES = ['Anas_platyrhynchos', 'Anolis_carolinensis', 'Bos_taurus', 'Callorhinchus_milii',
                  'Monodelphis_domestica', 'Ornithorhynchus_anatinus', 'Xenopus_tropicalis']

CHUNK_NUM = 1000000

ProgressBar().register()

def create_V(organism_names: list) -> da.Array:
    V = None
    for i in range(len(organism_names)):
        print(f'Processing organism {organism_names[i]}...')
        for j in range(len(organism_names)):
            DIR = f'/home/mashkova/ortologs/{organism_names[j]}'
            filename = os.path.join(DIR, f'{organism_names[i]}_{organism_names[j]}.out')
            if V is None:
                V = dd.read_csv(filename, delimiter="\t", header=None, 
                                names=['qseqid', 'sseqid', 'score', 'evalue'])
                V['sseqid'] = V['sseqid'].apply(lambda x: x.split('|')[1], meta=("sseqid", "object"))
            else:
                df = dd.read_csv(filename, delimiter="\t", header=None, 
                                 names=['qseqid', 'sseqid', 'score', 'evalue'])
                df['sseqid'] = df['sseqid'].apply(lambda x: x.split('|')[1], meta=("sseqid", "object"))
                V = dd.concat([V, df])
    V = V.repartition(256)
    print('Computing V matrix...')
    V = V.compute()
    print('Encoding protein names...')
    le = LabelEncoder()
    le.fit(np.unique(V[['qseqid', 'sseqid']].values.flatten()))
    V[['qseqid', 'sseqid']] = V[['qseqid', 'sseqid']].apply(le.fit_transform)
    V = V.values.astype('float64')
    print('Done!')
    return V, le.classes_


V, protein_names = create_V(ORGANISM_NAMES)
print(f'V matrix has size {V.shape}')

print('Saving protein names to .npz file...')
np.savez('/home/mashkova/ortologs/7_species_res/protein_names_V', x=protein_names)
dasked_V = da.from_array(V, chunks=CHUNK_NUM)

print('Saving V matrix to .hdf5 file...')
dasked_V.to_hdf5('/home/mashkova/ortologs/7_species_res/V.hdf5', '/x')
print('Done!')
