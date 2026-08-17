import numpy as np
import torch
from torch.utils import data



class DataSetBase(data.Dataset):
    def __init__(self, data, label, neighbors_index, transform=None, name=''):
        self.data = data
        self.label = label
        self.neighbors_index = neighbors_index
        self.data_name = name
        self.transform = transform

    def __len__(self):
        return len(self.data)


    def augment(self, data_input_item, context):
        
        index = context["index"]
        uniform_param = 1
        augment_bool = True
        neighbors_index = context["m_neighbors"]
        data_all = context["data_all"]
        
        if augment_bool:
            neighbor_index_list = neighbors_index[index]
            selected_index = np.random.choice(neighbor_index_list)
            alpha = np.random.uniform(0, uniform_param)
            try:
                selected_data_input = data_all[selected_index]
                data_input_aug = data_input_item * alpha + selected_data_input * (1 - alpha)
            except:
                data_input_aug = data_input_item
        else:
            data_input_aug = data_input_item
        return data_input_item, data_input_aug

    def __getitem__(self, index):
        
        index = index % self.data.shape[0]

        # print('index', index)
        data_input_item = self.data[index]
        label = self.label[index]
        
        context={
            "index": index,
            "m_neighbors": self.neighbors_index,
            "data_all": self.data,
        }
        data_input_item, data_input_aug = self.transform(x=data_input_item, context=context)
        # data_input_item, data_input_aug = self.augment(data_input_item, context)
        
        data_input_item = data_input_item.astype(np.float32)
        data_input_aug = data_input_aug.astype(np.float32)
        
        # print('data_input_item', data_input_item.shape, 'data_input_aug', data_input_aug.shape)
        return {
            "data_input_item": data_input_item,
            "data_input_aug": data_input_aug,
            "label": label,
            "index": index,
        }