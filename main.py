from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from model import Model, train, test


from data_loader import RawDataLoader
from evaluation import Evaluation
from utils import *
import random
import torch
import numpy as np
import pandas as pd

batch_size = 64

ae_latent_dim = 50
num_epochs = 25

def train_Model(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, cell_sizes, drug_sizes,device):
   
    model = Model(cell_sizes, drug_sizes, ae_latent_dim, ae_latent_dim)
    model= model.to(device)

    x_cell_train_tensor = torch.Tensor(x_cell_train.values)
    x_drug_train_tensor = torch.Tensor(x_drug_train.values)
    x_cell_train_tensor = torch.nn.functional.normalize(x_cell_train_tensor, dim=0)
    x_drug_train_tensor = torch.nn.functional.normalize(x_drug_train_tensor, dim=0)
    y_train_tensor = torch.Tensor(y_train)
    y_train_tensor = y_train_tensor.unsqueeze(1)


    classes = np.array([0, 1]) 
    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=classes, y=y_train),
                                 dtype=torch.float32)


    x_cell_train_tensor, x_cell_val_tensor, x_drug_train_tensor, x_drug_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
        x_cell_train_tensor, x_drug_train_tensor, y_train_tensor, test_size=0.1,
        random_state=RANDOM_SEED,
        shuffle=True)


    train_dataset = TensorDataset(x_cell_train_tensor.to(device), x_drug_train_tensor.to(device), y_train_tensor.to(device))
    val_dataset = TensorDataset(x_cell_val_tensor.to(device), x_drug_val_tensor.to(device), y_val_tensor.to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    train(model, train_loader, val_loader, num_epochs,class_weights)

    torch.save(model.state_dict(), 'MODEL.pth')

    model = Model(cell_sizes, drug_sizes, ae_latent_dim, ae_latent_dim) 
    model.load_state_dict(torch.load('MODEL.pth', weights_only=True))   
    model = model.to(device)

    x_cell_test_tensor = torch.Tensor(x_cell_test.values)
    x_drug_test_tensor = torch.Tensor(x_drug_test.values)
    y_test_tensor = torch.Tensor(y_test).to(device)

    x_cell_test_tensor = torch.nn.functional.normalize(x_cell_test_tensor, dim=0).to(device)
    x_drug_test_tensor = torch.nn.functional.normalize(x_drug_test_tensor, dim=0).to(device)

    test_dataset = TensorDataset(x_cell_test_tensor, x_drug_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=len(x_cell_test))

    return test(model, test_loader)

def cv_train(x_cell_train, x_drug_train, y_train, cell_sizes,
                                    drug_sizes, device, k=5, ):


    splits = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(x_cell_train)))):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        model = Model(cell_sizes, drug_sizes, ae_latent_dim, ae_latent_dim)

        x_cell_train_tensor = torch.Tensor(x_cell_train.values)
        x_drug_train_tensor = torch.Tensor(x_drug_train.values)

        y_train_tensor = torch.Tensor(y_train)
        y_train_tensor = y_train_tensor.unsqueeze(1)

        classes = np.array([0, 1]) 
        class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=classes, y=y_train),
                                     dtype=torch.float32)

        train_dataset = TensorDataset(x_cell_train_tensor, x_drug_train_tensor, y_train_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        train(model, train_loader,train_loader, num_epochs, class_weights)


        test_loader = DataLoader(train_dataset, batch_size=len(x_cell_train), sampler=test_sampler)

        results = test(model, test_loader)

        Evaluation.add_results(history, results)


    return Evaluation.show_final_results(history)

def run(k, is_test=False ):
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}

    train_data, train_drug_screen = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                            raw_file_directory=GDSC_RAW_DATA_FOLDER,
                                                            screen_file_directory=GDSC_SCREENING_DATA_FOLDER,
                                                            sep="\t")


    if is_test:
        test_data, test_drug_screen = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                              raw_file_directory=CCLE_RAW_DATA_FOLDER,
                                                              screen_file_directory=CCLE_SCREENING_DATA_FOLDER,
                                                              sep="\t")
        train_data, test_data = RawDataLoader.data_features_intersect(train_data, test_data)

       

    x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(train_data,
                                                                                                    train_drug_screen)
    
    # print("x_drug_train shape:", x_drug_train.shape)
    # print("x_cell_train shape:", x_cell_train.shape)

    if is_test:
        x_cell_test, x_drug_test, y_test, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(test_data,
                                                                                                    test_drug_screen)

    rus = RandomUnderSampler(sampling_strategy="majority", random_state=RANDOM_SEED)
    dataset = pd.concat([x_cell_train, x_drug_train], axis=1)
    dataset.index = x_cell_train.index
    dataset, y_train = rus.fit_resample(dataset, y_train)
    x_cell_train = dataset.iloc[:, :sum(cell_sizes)]
    x_drug_train = dataset.iloc[:, sum(cell_sizes):]

    for i in range(k):
        print('Run {}'.format(i))

        if is_test:

            results = train_Model(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, cell_sizes,
                                    drug_sizes, device)

        else:
            
            results = cv_train(x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes, device, k=5)

        Evaluation.add_results(history, results)

    Evaluation.show_final_results(history)
    return history


if __name__ == '__main__':

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    run(1, is_test=True)
        
