from sklearn.datasets import load_digits

from torch.utils import data
import numpy as np
import os
from sklearn.decomposition import PCA
from pynndescent import NNDescent
from sklearn.metrics import pairwise_distances
import joblib

from data_model.aug import augment_base as augment




class DigitsDataset(data.Dataset):
    def __init__(
        self,
        data_name="Digits",
        data_path="/zangzelin/data",
        k=10,
        pca_dim=100,
        uselabel=False,
        uniform_param=1,
        train_val="train",
        split_ratio=1,
        split_ratio_test=0,
        num_positive_samples=1,
        use_neighbor=True,
        augment_bool=True,
        len_multiple=50,
        transform=None,
    ):
        self.data_name = data_name
        self.len_multiple = len_multiple
        data, label = self.load_data(data_path=data_path)
        self.transform = transform

        index_save_path = f'data/'
        

        

        #train_data, train_label, val_data, val_label, test_data, test_label = self.split_data(
        #    data, label,
        #    split_ratio=split_ratio,
        #    split_ratio_test=split_ratio_test,
        #    train_val=train_val,
        #    data_name=data_name,
        #    index_save_path=index_save_path,
        #    )
        
        if data_name == 'Cifar10' or data_name == 'Cifar10T10':
            use_neighbor = False
    
        #print('train_data.shape', train_data.shape, 
        #      'train_label.shape', train_label.shape, 
        #      'val_data.shape', val_data.shape, 
        #      'val_label.shape', val_label.shape,
        #      'test_data.shape', test_data.shape, 
        #      'test_label.shape', test_label.shape)
        
        print('train_val', train_val)
        
        self.train_val = train_val
        
        if train_val == 'train':
            self.data = data
            self.label = label
        elif train_val == 'val':
            self.data = data
            self.label = label
        elif train_val == 'test':
            self.data = data
            self.label = label
        
        self.uniform_param = uniform_param
        
        if use_neighbor:
            self.neighbors_index = self.Preprocessing(
                data, k, pca_dim, uselabel, train_val, split_ratio, split_ratio_test
            )
        
    def split_data(self, data, label, split_ratio=1, split_ratio_test=0, train_val='train', data_name='',index_save_path='data', index_save_name='train_val.npy'):

        data_len = data.shape[0]
        train_len = int(data_len * split_ratio)
        test_len = int(data_len * split_ratio_test)
        if not os.path.exists(index_save_path):
            os.makedirs(index_save_path)

        index_save_name_train = f'trainval_index_train_{split_ratio}_{split_ratio_test}_{data_name}.npy'
        index_save_name_val = f'trainval_index_val_{split_ratio}_{split_ratio_test}_{data_name}.npy'
        index_save_name_test = f'trainval_index_test_{split_ratio}_{split_ratio_test}_{data_name}.npy'
        if not os.path.exists(os.path.join(index_save_path, index_save_name_train)):
            print('save index', index_save_name_train, index_save_name_val, index_save_name_test)
            rand_index = np.random.permutation(data_len)
            train_index = rand_index[:train_len]
            val_index = rand_index[train_len: train_len + test_len]
            test_index = rand_index[train_len + test_len:]
            np.save(os.path.join(index_save_path, index_save_name_train), train_index)
            np.save(os.path.join(index_save_path, index_save_name_val), val_index)
            np.save(os.path.join(index_save_path, index_save_name_test), test_index)
        else:
            print(
                'load index', index_save_name_train, index_save_name_val, index_save_name_test,
                'from', os.path.join(index_save_path, index_save_name_train))
            train_index = np.load(os.path.join(index_save_path, index_save_name_train))
            val_index = np.load(os.path.join(index_save_path, index_save_name_val))
            test_index = np.load(os.path.join(index_save_path, index_save_name_test))
        train_data = data[train_index]
        train_label = label[train_index]
        val_data = data[val_index]
        val_label = label[val_index]
        test_data = data[test_index]
        test_label = label[test_index]

        return train_data, train_label, val_data, val_label , test_data, test_label

    def load_data(self, data_path):
        digit = load_digits()
        data = np.array(digit.data).astype(np.float32)
        label = np.array(digit.target)
        return data, label

    def Preprocessing(self, data, k, pca_dim, uselabel, token_index=0, train_val='train', split_ratio=1,split_ratio_test=0):
        self.graphwithpca = False

        # index = NNDescent(self.loc, n_jobs=-1)
        # neighbors_loc_index, neighbors_loc_dist = index.query(self.loc, k=k + 1)
        # neighbors_loc_index = neighbors_loc_index[:, 1:]
        
        # import pdb; pdb.set_trace()

        neighbors_index_gene = self.cal_near_index(
            k=k,
            uselabel=uselabel,
            pca_dim=pca_dim,
            train_val=train_val,
            split_ratio=split_ratio,
        )
        
        # neighbors_index = np.concatenate([neighbors_loc_index, neighbors_index_gene], axis=1)
        
        return neighbors_index_gene



    def cal_near_index(self, k=10, device="cuda", uselabel=False, pca_dim=100, train_val='train', split_ratio=1):
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}train_val{}split_ratio{}.pkl".format(
            self.data_name, k, uselabel, pca_dim, train_val, split_ratio
        )
        
        os.makedirs("save_near_index", exist_ok=True)
        if not os.path.exists(filename):
            X_rshaped = self.data.reshape((self.data.shape[0], -1))
            if pca_dim < X_rshaped.shape[1]:
                X_rshaped = PCA(n_components=pca_dim).fit_transform(X_rshaped)
            if not uselabel:
                index = NNDescent(X_rshaped, n_jobs=-1)
                neighbors_index, neighbors_dist = index.query(X_rshaped, k=k + 1)
                neighbors_index = neighbors_index[:, 1:]
            else:
                dis = pairwise_distances(X_rshaped)
                M = np.repeat(self.label.reshape(1, -1), X_rshaped.shape[0], axis=0)
                dis[(M - M.T) != 0] = dis.max() + 1
                neighbors_index = dis.argsort(axis=1)[:, 1 : k + 1]
            joblib.dump(value=neighbors_index, filename=filename)

            print("save data to ", filename)
        else:
            print("load data from ", filename)
            neighbors_index = joblib.load(filename)
        return neighbors_index

    def augment(self, data_input_item, index):
        
        # data_input_item, data_input_aug = augment(
        #     data_ori=data_input_item,
        #     index_ori=index,
        #     m_neighbors=self.neighbors_index,
        #     data_all=self.data,
        #     uniform_param=self.uniform_param,
        #     augment_bool=self.augment_bool
        # )

        # # print('------', index)
        # if self.augment_bool:
        #     neighbor_index_list = self.neighbors_index[index]
        #     selected_index = np.random.choice(neighbor_index_list)
        #     alpha = np.random.uniform(0, self.uniform_param)
        #     try:
        #         selected_data_input = self.data[selected_index]
        #         data_input_aug = data_input_item * alpha + selected_data_input * (1 - alpha)
        #     except:
        #         data_input_aug = data_input_item
        # else:
        #     # print('no augment', 'self.augment_bool', self.augment_bool, )
        #     data_input_aug = data_input_item

        return data_input_item, data_input_aug

    def __getitem__(self, index):
        
        index = index % self.data.shape[0]

        # print('index', index)
        data_input_item = self.data[index]
        label = self.label[index]
        data_input_item, data_input_aug = self.augment(data_input_item, index)
        
        data_input_item = data_input_item.astype(np.float32)
        data_input_aug = data_input_aug.astype(np.float32)
        
        # print('data_input_item', data_input_item.shape, 'data_input_aug', data_input_aug.shape)
        return {
            "data_input_item": data_input_item,
            "data_input_aug": data_input_aug,
            "label": label,
            "index": index,
        }

    def __len__(self):
        # if 'Test' not in self.data_name:
        #     return self.data.shape[0] * 50
        # else:
        #     return self.data.shape[0]
        return self.data.shape[0] * self.len_multiple

