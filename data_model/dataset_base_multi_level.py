import numpy as np
import torch
from torch.utils import data



class DataSetBaseMultiLevel(data.Dataset):
    def __init__(self, data, label, neighbors_index, transform=None, name=''):
        self.data = data
        self.label = label
        self.neighbors_index = neighbors_index
        self.data_name = name
        self.transform = transform
        
        # self.num_levels = len(data_list)
        # self.data_list = data_list
        # self.label_list = label_list
        # self.neighbors_index_list = neighbors_index_list

    def __len__(self):
        return len(self.data)


    # def _get_item_from_level(self, index, level):

    #     index = index % self.data_list[level].shape[0]

    #     # print('index', index)
    #     data_input_item = self.data_list[level][index]

    #     context={
    #         "index": index,
    #         "m_neighbors": self.neighbors_index_list[level],
    #         "data_all": self.data_list[level],
    #         "label": self.label_list[level],
    #     }
    #     transform = self.transform[level]
    #     data_input_item, data_input_aug = transform(x=data_input_item, context=context)
    #     # data_input_item, data_input_aug = self.augment(data_input_item, context)
        
    #     data_input_item = data_input_item.astype(np.float32)
    #     data_input_aug = data_input_aug.astype(np.float32)

    #     return data_input_item, data_input_aug


    def _get_item(self, index):

        index = index % self.data.shape[0]

        # print('index', index)
        data_input_item = self.data[index]

        context={
            "index": index,
            "m_neighbors": self.neighbors_index,
            "data_all": self.data,
            "label": self.label,
        }
        transform = self.transform
        data_input_item, data_input_aug = transform(x=data_input_item, context=context)
        # data_input_item, data_input_aug = self.augment(data_input_item, context)
        
        data_input_item = data_input_item.astype(np.float32)
        data_input_aug = data_input_aug.astype(np.float32)

        return data_input_item, data_input_aug

    def __getitem__(self, index):
        
        label = self.label[index]
        # data_input_item_list = []
        # data_input_aug_list = []
        
        data_input_item, data_input_aug = self._get_item(index)
        
        
        # for level in range(self.num_levels):
            # data_input_item, data_input_aug = self._get_item_from_level(index, level)
            # data_input_item_list.append(data_input_item)
            # data_input_aug_list.append(data_input_aug)
        
        # data_input_item_list = np.array(data_input_item_list)
        # data_input_aug_list = np.array(data_input_aug_list)
        
        # print('data_input_item', data_input_item.shape, 'data_input_aug', data_input_aug.shape)
        return {
            "data_input_item": data_input_item,
            "data_input_aug": data_input_aug,
            # "data_input_item_list": data_input_item_list,
            # "data_input_aug_list": data_input_aug_list,
            "label": label,
            "index": index,
        }