from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

loc_str = '/cluster/home/sagraw/acad_influence/CitationByYearData/processed6/'

def get_data(batch_train, batch_test):
    train_data_nts   = np.load(loc_str + 'trainx_nts.npy')
    train_data_abs   = np.load(loc_str + 'trainx_abs.npy')
    train_data_ts   = np.load(loc_str + 'trainx_ts.npy')
    train_labels = np.load(loc_str + 'trainy.npy')
    valid_data_nts   = np.load(loc_str + 'validx_nts.npy')
    valid_data_abs   = np.load(loc_str + 'validx_abs.npy')
    valid_data_ts   = np.load(loc_str + 'validx_ts.npy')
    valid_labels = np.load(loc_str + 'validy.npy')
    test_data_nts    = np.load(loc_str + 'testx_nts.npy')
    test_data_abs    = np.load(loc_str + 'testx_abs.npy')
    test_data_ts    = np.load(loc_str + 'testx_ts.npy')
    test_labels  = np.load(loc_str + 'testy.npy')

    ## Train data:
    train_dataset = TensorDataset(Tensor(train_data_nts).float(),Tensor(train_data_abs).float(),Tensor(train_data_ts).float(), Tensor(train_labels).float())
    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_train)

    ## Test data
    test_dataset = TensorDataset(Tensor(test_data_nts).float(),Tensor(test_data_abs).float(),Tensor(test_data_ts).float(), Tensor(test_labels).float())
    testloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_test)

    ## Valid data
    valid_dataset = TensorDataset(Tensor(valid_data_nts).float(),Tensor(valid_data_abs).float(),Tensor(valid_data_ts).float(), Tensor(valid_labels).float())
    validloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_test)

    return trainloader, train_dataset, validloader, valid_dataset, testloader, test_dataset
