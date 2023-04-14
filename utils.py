import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader


class VAEDataset(Dataset):
    def __init__(self, x, c_cols, b_cols):
        x_c = x.loc[:, c_cols]
        x_b = x.loc[:, b_cols]
        x_c = [torch.from_numpy(v.values) for _, v in x_c.groupby(level=0)]
        x_b = [torch.from_numpy(v.values) for _, v in x_b.groupby(level=0)]
        self.m_c = [~torch.isnan(v) for v in x_c]
        self.m_b = [~torch.isnan(v) for v in x_b]
        self.x_c = [torch.nan_to_num(v, 0) for v in x_c]
        self.x_b = [torch.nan_to_num(v, 0) for v in x_b]

    def __len__(self):
        return len(self.x_c)

    def __getitem__(self, idx):
        return self.x_c[idx], self.m_c[idx],\
               self.x_b[idx], self.m_b[idx]


def cfn(batch):
    x_c, m_c, x_b, m_b = zip(*batch)

    batch_size = len(batch)
    cfn_lengths = torch.Tensor(list(map(lambda x: x.shape[0], x_c))).long()
    max_length = int(cfn_lengths.max())

    x_c_sz = x_c[0].shape[1]
    x_b_sz = x_b[0].shape[1]
    cfn_x_c = torch.zeros([batch_size, max_length, x_c_sz])
    cfn_m_c = torch.zeros([batch_size, max_length, x_c_sz])
    cfn_x_b = torch.zeros([batch_size, max_length, x_b_sz])
    cfn_m_b = torch.zeros([batch_size, max_length, x_b_sz])

    for i in range(batch_size):
        cfn_x_c[i, -int(cfn_lengths[i]):, :] = x_c[i]
        cfn_m_c[i, -int(cfn_lengths[i]):, :] = m_c[i]
        cfn_x_b[i, -int(cfn_lengths[i]):, :] = x_b[i]
        cfn_m_b[i, -int(cfn_lengths[i]):, :] = m_b[i]

    return cfn_x_c, cfn_m_c, cfn_x_b, cfn_m_b, cfn_lengths


def transform(data, c_cols):
    scaler = StandardScaler()
    data.loc[:, c_cols] = scaler.fit_transform(data.loc[:, c_cols])
    data_min = data.min(axis=0)
    data = data - data_min
    return data


def inverse_transform(data, data_min, scaler):
    data = data + data_min
    data.loc[:, c_cols] = scaler.inverse_transform()
    pass


def prepare_data(args):
    data = pd.read_csv(args.data_path, index_col=[0, 1])
    data = data.loc[:, args.c_cols + args.b_cols]
    data = transform(data, args.c_cols)

    # train, test data
    ids = data.index.get_level_values(0).unique()
    index_train, index_test = train_test_split(ids, test_size=.1, random_state=args.seed)
    data_train = data.loc[sorted(index_train), :]
    data_test = data.loc[sorted(index_test), :]
    dataset_test = VAEDataset(data_test, args.c_cols, args.b_cols)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test.x_b), shuffle=False, collate_fn=cfn)

    # cross-validation data
    ids_train = data_train.index.get_level_values(0).unique()
    kf = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    data_cv = []
    for index_train, index_val in kf.split(ids_train):
        data_t = data_train.loc[sorted(ids_train[index_train]), :]
        data_v = data_train.loc[sorted(ids_train[index_val]), :]

        dataset_t = VAEDataset(data_t, args.c_cols, args.b_cols)
        dataset_v = VAEDataset(data_v, args.c_cols, args.b_cols)

        dataloader_t = DataLoader(dataset_t, batch_size=args.batch_size, shuffle=True, collate_fn=cfn)
        dataloader_v = DataLoader(dataset_v, batch_size=len(dataset_v.x_b), shuffle=False, collate_fn=cfn)

        data_cv.append({"data_train": data_t,
                        "data_val": data_v,
                        "dataloader_train": dataloader_t,
                        "dataloader_val": dataloader_v})

    data = {"data_cv": data_cv,
            "data_test": {"data_test": data_test,
                          "dataloader_test": dataloader_test}}
    return data


def kl_anneal_function(epoch, shape, pos, max_value):
    return (max_value/(1+np.exp(-shape*(epoch-pos))))

