import os


DATA_FOLDER = 'data'
DRUG_DATA_FOLDER = os.path.join(DATA_FOLDER, 'drug_data')
GDSC_RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'GDSC_data')
CCLE_RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'CCLE_data')

GDSC_SCREENING_DATA_FOLDER = os.path.join(GDSC_RAW_DATA_FOLDER, 'drug_screening_matrix_GDSC.tsv')
CCLE_SCREENING_DATA_FOLDER = os.path.join(CCLE_RAW_DATA_FOLDER, 'drug_screening_matrix_ccle.tsv')


GDSC_FOLDER = os.path.join(DATA_FOLDER, 'GDSC')
CCLE_FOLDER = os.path.join(DATA_FOLDER, 'CCLE')

MODEL_FOLDER = os.path.join(DATA_FOLDER, 'model')



BUILD_SIM_MATRICES = True  
SIM_KERNEL = {'cell_CN': ('euclidean', 0.001), 'cell_exp': ('euclidean', 0.01), 'cell_methy': ('euclidean', 0.1),
              'cell_mut': ('jaccard', 1), 'drug_DT': ('jaccard', 1), 'drug_comp': ('euclidean', 0.001),
              'drug_desc': ('euclidean', 0.001), 'drug_finger': ('euclidean', 0.001)}
SAVE_MODEL = False  # Change it to True to save the trained model
VARIATIONAL_AUTOENCODERS = False
# DATA_MODALITIES = ['cell_CN', 'cell_exp', 'cell_mut', 'drug_desc', 'drug_finger']
DATA_MODALITIES = ['cell_exp', 'drug_desc','drug_finger']

RANDOM_SEED = 58
deterministic = True

DRUG_SMILES_FILE = os.path.join(DRUG_DATA_FOLDER, 'drug_names_with_smiles.csv')
