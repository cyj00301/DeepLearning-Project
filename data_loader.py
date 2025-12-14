import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.impute import SimpleImputer
import os

from utils import DRUG_DATA_FOLDER


class RawDataLoader:
    @staticmethod
    def load_data(data_modalities, raw_file_directory, screen_file_directory, sep, drug_directory=DRUG_DATA_FOLDER):
        
        data = RawDataLoader.load_raw_files(intersect=True, data_modalities=data_modalities,
                                            raw_file_directory=raw_file_directory)

        drug_data = RawDataLoader.load_raw_files(intersect=True, data_modalities=data_modalities,
                                                 raw_file_directory=drug_directory)

        data.update(drug_data)

        drug_screen = RawDataLoader.load_screening_files(
            filename=screen_file_directory,
            sep=sep)
        drug_screen, data = RawDataLoader.adjust_screening_raw(
            drug_screen=drug_screen, data_dict=data)

        return data, drug_screen

    @staticmethod
    def intersect_features(data1, data2):
        common_columns = [c for c in data1.columns if c in data2.columns]

        data1 = data1[common_columns]
        data2 = data2[common_columns]

        return data1, data2

    @staticmethod
    def data_features_intersect(data1, data2):
        
        for i in data1:
            data1[i], data2[i] = RawDataLoader.intersect_features(data1[i], data2[i])
        return data1, data2

    @staticmethod
    def load_file(address, index_column=None):
        
        data = []
        try:
            if address.endswith('.txt') or address.endswith('.tsv'):
                data.append(pd.read_csv(address, sep='\t', index_col=index_column), )
            elif address.endswith('.csv'):
                data.append(pd.read_csv(address))
            elif address.endswith('.xlsx'):
                data.append(pd.read_excel(address))
        except FileNotFoundError:
            print(f'File not found at address: {address}')

        return data[0]

    @staticmethod
    def load_raw_files(raw_file_directory, data_modalities, intersect=True):
        raw_dict = {}
        files = os.listdir(raw_file_directory)
        cell_line_names = None
        drug_names = None
        for file in tqdm(files, 'Reading Raw Data Files...'): 
            if any([file.startswith(x) for x in data_modalities]):
                if file.endswith('_raw.gzip'):
                    df = pd.read_parquet(os.path.join(raw_file_directory, file))
                elif file.endswith('_raw.tsv'):
                    df = pd.read_csv(os.path.join(raw_file_directory, file), sep='\t', index_col=0)
                else:
                    continue
                if df.index.is_numeric():
                    df = df.set_index(df.columns[0])
                df = df.sort_index()
                df = df.sort_index(axis=1)
                df.columns = df.columns.str.replace('_cell_mut', '')
                df.columns = df.columns.str.replace('_cell_CN', '')
                df.columns = df.columns.str.replace('_cell_exp', '')
                df = (df - df.min()) / (df.max() - df.min())
                df = df.fillna(0)
                if intersect:
                    if file.startswith('cell'):
                        if cell_line_names:
                            cell_line_names = cell_line_names.intersection(set(df.index))
                        else:
                            cell_line_names = set(df.index)
                    elif file.startswith('drug'):
                        if drug_names:
                            drug_names = drug_names.intersection(set(df.index))
                        else:
                            drug_names = set(df.index)
                raw_dict[file[:file.find('_raw')]] = df
       
        if cell_line_names is not None:
            set_cell = frozenset(cell_line_names)
            cell_line_names = [x for x in df.index if x in set_cell]

        if drug_names is not None:
            set_drug = frozenset(drug_names)
            drug_names = [x for x in df.index if x in set_drug]
            
        if intersect:
            for key, value in raw_dict.items():
                if key.startswith('cell'):
                    data = value.loc[list(cell_line_names)]
                    raw_dict[key] = data.loc[~data.index.duplicated()]
                elif key.startswith('drug'):
                    data = value.loc[list(drug_names)]
                    raw_dict[key] = data.loc[~data.index.duplicated()]

        return raw_dict

    @staticmethod
    def load_screening_files(filename="AUC_matS_comb.tsv", sep=',', ):
        df = pd.read_csv(filename, sep=sep, index_col=0)
        return df

    @staticmethod
    def adjust_screening_raw(drug_screen, data_dict):
        raw_cell_names = []
        raw_drug_names = []
        for key, value in data_dict.items():
            if 'cell' in key:
                if len(raw_cell_names) == 0:
                    raw_cell_names = value.index
                else:
                    raw_cell_names = raw_cell_names.intersection(value.index)
            elif 'drug' in key:
                raw_drug_names = value.index

        screening_cell_names = drug_screen.index
        screening_drug_names = drug_screen.columns

        # common_cell_names = list(set(raw_cell_names).intersection(set(screening_cell_names)))
        # common_drug_names = list(set(raw_drug_names).intersection(set(screening_drug_names)))
       
        set_cell = frozenset(screening_cell_names)
        set_drug = frozenset(screening_drug_names)
        common_cell_names = [x for x in raw_cell_names if x in set_cell]
        common_drug_names = [x for x in raw_drug_names if x in set_drug]

        for key, value in data_dict.items():
            if 'cell' in key:
                data_dict[key] = value.loc[common_cell_names]
            else:
                data_dict[key] = value.loc[common_drug_names]
        return drug_screen.loc[common_cell_names, common_drug_names], data_dict

       
    @staticmethod
    def prepare_input_data(data_dict, screening):
        print('Preparing data...')
        resistance = np.argwhere((screening.to_numpy() == 1)).tolist()
        resistance.sort(key=lambda x: (x[1], x[0]))
        resistance = np.array(resistance)
        sensitive = np.argwhere((screening.to_numpy() == -1)).tolist()
        sensitive.sort(key=lambda x: (x[1], x[0]))
        sensitive = np.array(sensitive)

        # print("sensitive train data len:", len(sensitive))
        # print("resistance train data len:", len(resistance))

        A_train_mask = np.ones(len(resistance), dtype=bool)
        B_train_mask = np.ones(len(sensitive), dtype=bool)
        resistance = resistance[A_train_mask]
        sensitive = sensitive[B_train_mask]
        cell_data_types = list(filter(lambda x: x.startswith('cell'), data_dict.keys()))
        cell_data_types.sort()
        cell_data = pd.concat(
            [pd.DataFrame(data_dict[data_type].add_suffix(f'_{data_type}'), dtype=np.float32) for
             data_type in cell_data_types], axis=1)
        cell_data_sizes = [data_dict[data_type].shape[1] for data_type in cell_data_types]

        drug_data_types = list(filter(lambda x: x.startswith('drug'), data_dict.keys()))
        drug_data_types.sort()
        drug_data = pd.concat(
            [pd.DataFrame(data_dict[data_type].add_suffix(f'_{data_type}'), dtype=np.float32, )
             for data_type in drug_data_types], axis=1)
        drug_data_sizes = [data_dict[data_type].shape[1] for data_type in drug_data_types]

        Xp_cell = cell_data.iloc[resistance[:, 0], :]
        Xp_drug = drug_data.iloc[resistance[:, 1], :]
        Xp_cell = Xp_cell.reset_index(drop=True)
        Xp_drug = Xp_drug.reset_index(drop=True)
        Xp_cell.index = [f'({screening.index[x[0]]},{screening.columns[x[1]]})' for x in resistance]
        Xp_drug.index = [f'({screening.index[x[0]]},{screening.columns[x[1]]})' for x in resistance]

        Xn_cell = cell_data.iloc[sensitive[:, 0], :]
        Xn_drug = drug_data.iloc[sensitive[:, 1], :]
        Xn_cell = Xn_cell.reset_index(drop=True)
        Xn_drug = Xn_drug.reset_index(drop=True)
        Xn_cell.index = [f'({screening.index[x[0]]},{screening.columns[x[1]]})' for x in sensitive]
        Xn_drug.index = [f'({screening.index[x[0]]},{screening.columns[x[1]]})' for x in sensitive]

        X_cell = pd.concat([Xp_cell, Xn_cell])
        X_drug = pd.concat([Xp_drug, Xn_drug])

        Y = np.append(np.zeros(resistance.shape[0]), np.ones(sensitive.shape[0]))
        return X_cell, X_drug, Y, cell_data_sizes, drug_data_sizes
