import dask.dataframe as dd
import dask.array as da
from dask import compute
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import LabelEncoder
from dask.diagnostics import ProgressBar
import gffpandas.gffpandas as gffpd


#ORGANISM_NAMES = ['Anas_platyrhynchos', 'Anolis_carolinensis', 'Bos_taurus', 'Callorhinchus_milii',
#                  'Monodelphis_domestica', 'Ornithorhynchus_anatinus', 'Xenopus_tropicalis']
#ORGANISM_IDS = ['GCF_003850225.1', 'GCF_000090745.1', 'GCF_002263795.2', 'GCF_000165045.1',
#                'GCF_000002295.2', 'GCF_004115215.2', 'GCF_000004195.4']

ORGANISM_NAMES = ['Monodelphis_domestica', 'Xenopus_tropicalis']
ORGANISM_IDS = ['GCF_000002295.2', 'GCF_000004195.4']

#ORGANISM_NAMES = ['Anas_platyrhynchos', 'Anolis_carolinensis', 'Bos_taurus', 'Callorhinchus_milii',
#                  'Canis_lupus_familiaris', 'Chrysemys_picta_bellii', 'Crocodylus_porosus', 'Danio_rerio',
#                  'Equus_caballus', 'Felis_catus', 'Gallus_gallus', 'Latimeria_chalumnae',
#                  'Meleagris_gallopavo', 'Monodelphis_domestica', 'Notechis_scutatus',
#                  'Ornithorhynchus_anatinus', 'Taeniopygia_guttata', 'Xenopus_tropicalis']
#ORGANISM_IDS = ['GCF_003850225.1', 'GCF_000090745.1', 'GCF_002263795.2', 'GCF_000165045.1',
#                'GCF_000002285.5', 'GCF_000241765.4', 'GCF_001723895.1', 'GCF_000002035.6',
#                'GCF_002863925.1', 'GCF_000181335.3', 'GCF_000002315.6', 'GCF_000225785.1',
#                'GCF_000146605.3', 'GCF_000002295.2', 'GCF_900518725.1',
#                'GCF_004115215.2', 'GCF_003957565.2', 'GCF_000004195.4']

CHUNK_NUM = 1000000

ProgressBar().register()

def read_gffs(organism_names: list, organism_ids: list) -> pd.DataFrame:
    gff_df = pd.DataFrame()
    for i in range(len(organism_names)):
        print(f'Processing organism {organism_names[i]}...')
        fpath = f'{organism_names[i]}/ncbi_dataset/data/{organism_ids[i]}/genomic.gff'
        annotation = gffpd.read_gff3(fpath)
        genes = annotation.filter_feature_of_type(['CDS']).attributes_to_columns()
        genes['species'] = [organism_names[i]] * genes.shape[0]
        gff_df = pd.concat([gff_df, genes])
    gff_df = gff_df[gff_df['gene'].notna()]
    gff_df = gff_df[gff_df['protein_id'].notna()]
    gff_df = gff_df[['protein_id', 'gene', 'species']]
    gff_df.drop_duplicates(ignore_index=True, inplace=True)
    gff_df.reset_index(drop=True, inplace=True)
    return gff_df

print('Reading .gff files...')
gff_df = read_gffs(ORGANISM_NAMES, ORGANISM_IDS)
print('Done!')

print('Reading norm_sym_V matrix from .hdf5 file...')
f = h5py.File('/home/mashkova/ortologs/2_species_res/norm_sym_V.hdf5')
d = f['/x']
norm_V_da = da.from_array(d, chunks=CHUNK_NUM)
norm_V_df = dd.from_dask_array(norm_V_da, columns=['qseqid', 'sseqid', 'score', 'evalue'])

print('Reading protein names from .npz file...')
protein_names = np.load('/home/mashkova/ortologs/2_species_res/protein_names_norm_sym_V.npz', allow_pickle=True)['x']

print('Decoding protein names...')
norm_V_df['qseqid'] = norm_V_df['qseqid'].apply(lambda x: protein_names[int(x)], meta=('qseqid', 'object'))
norm_V_df['sseqid'] = norm_V_df['sseqid'].apply(lambda x: protein_names[int(x)], meta=('sseqid', 'object'))

print('Sorting norm_sym_V matrix by qseqid and sseqid...')
norm_V_df_qseqid = norm_V_df.compute().sort_values(['qseqid', 'sseqid'])

print('Sorting gff_df by protein_id and gene...')
gff_df = gff_df.sort_values(['protein_id', 'gene'])

print('Merging...')
df_ = gff_df.rename(columns={'protein_id': 'qseqid'})
df = pd.merge(norm_V_df_qseqid, df_, on='qseqid')
df = df[['qseqid', 'sseqid', 'score', 'evalue', 'gene', 'species']]
df.rename(columns={'gene': 'qseqid_gene', 'species':'qseqid_species'}, inplace=True)
df_ = gff_df.rename(columns={'protein_id': 'sseqid'})
df = pd.merge(df, df_, on='sseqid')
df = df[['qseqid', 'sseqid', 'score', 'evalue', 'qseqid_gene', 'qseqid_species', 'gene', 'species']]
df.rename(columns={'gene': 'sseqid_gene', 'species':'sseqid_species'}, inplace=True)
print(df)

print('Extracting max scores...')
df = df.loc[df.groupby(['qseqid_gene','sseqid_gene'])['score'].idxmax()].reset_index(drop=True)
df = df[['qseqid_gene', 'sseqid_gene', 'score', 'evalue', 'qseqid_species', 'sseqid_species']]

print('Encoding gene names...')
le = LabelEncoder()
le.fit(np.unique(df[['qseqid_gene', 'sseqid_gene']].values.flatten()))
df[['qseqid_gene', 'sseqid_gene']] = df[['qseqid_gene', 'sseqid_gene']].apply(le.transform)

print('Saving gene names to .npz file...')
np.savez('/home/mashkova/ortologs/2_species_res/gene_names_W', x=le.classes_)

print("Encoding species names...")
le = LabelEncoder()
le.fit(np.unique(df[['qseqid_species', 'sseqid_species']].values.flatten()))
df[['qseqid_species', 'sseqid_species']] = df[['qseqid_species', 'sseqid_species']].apply(le.transform)

print('Saving species names to .npz file...')
np.savez('/home/mashkova/ortologs/2_species_res/species_names_W', x=le.classes_)

print('Saving W matrix to .hdf5 file...')
W = df.values.astype('float64')
dasked_W = da.from_array(W, chunks=CHUNK_NUM)
dasked_W.to_hdf5('/home/mashkova/ortologs/2_species_res/W.hdf5', '/x')
print('Done!')
