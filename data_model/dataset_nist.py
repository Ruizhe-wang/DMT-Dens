from sklearn.datasets import load_digits

from torch.utils import data
import numpy as np
import os
from sklearn.decomposition import PCA
from pynndescent import NNDescent
from sklearn.metrics import pairwise_distances
import joblib
from PIL import Image

from data_model.dataset_meta import DigitsDataset
# from data_model.dataset_meta import DigitsSEQDataset
import torchvision.datasets as datasets
from PIL import Image

from torch.utils.data import TensorDataset, DataLoader, Subset, ConcatDataset
import torchvision.transforms as T

from sklearn.model_selection import train_test_split

from torchvision import transforms
import torchvision.transforms.functional as F
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

class MnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((60000, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((10000, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        
        data = data-data.mean()
        labels = np.concatenate((train_labels, test_labels))
        print("Mnist",data.shape)
        return data, labels



class MnistTestDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((60000, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((10000, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        
        data = data-data.mean()
        labels = np.concatenate((train_labels, test_labels))
        print("Mnist",data.shape)
        return data[:10000], labels[:10000]

class FasionMnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((60000, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((10000, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        
        data = data-data.mean()
        labels = np.concatenate((train_labels, test_labels))
        print("Mnist",data.shape)
        return data, labels

class Mnist10000Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((60000, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((10000, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        labels = np.concatenate((train_labels, test_labels))
        
        random_idx = np.random.permutation(data.shape[0])[:10000]
        data = data[random_idx]
        labels = labels[random_idx]
        
        print("Mnist",data.shape)
        return data, labels

class Mnist10000TestDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((60000, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((10000, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        labels = np.concatenate((train_labels, test_labels))
        
        random_idx = np.random.permutation(data.shape[0])[:10000]
        data = data[random_idx]
        labels = labels[random_idx]
        
        print("Mnist",data.shape)
        return data, labels

class FMnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=None)
        data = (np.array(D.data[:60000]).astype(np.float32) / 255).reshape((60000, -1))
        label = np.array(D.targets[:60000]).reshape((-1))
        return data, label

class KMnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.KMNIST(root=data_path, train=True, download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((60000, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.KMNIST(root=data_path, train=False, download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((10000, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        labels = np.concatenate((train_labels, test_labels))
        return data, labels

class EMnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.EMNIST(root=data_path, train=True, split="byclass", download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((697932, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.EMNIST(root=data_path, train=False, split="byclass", download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((116323, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        labels = np.concatenate((train_labels, test_labels))
        return data, labels

class EMnist18Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_set = datasets.EMNIST(root=data_path, train=True, split="byclass", download=True, transform=None)
        train_data = (np.array(train_set.data).astype(np.float32) / 255).reshape((697932, -1))
        train_labels = np.array(train_set.targets).reshape((-1))
        
        test_set = datasets.EMNIST(root=data_path, train=False, split="byclass", download=True, transform=None)
        test_data = (np.array(test_set.data).astype(np.float32) / 255).reshape((116323, -1))
        test_labels = np.array(test_set.targets).reshape((-1))
        
        data = np.vstack((train_data, test_data))
        labels = np.concatenate((train_labels, test_labels))
        return data[:180000], labels[:180000]

class Coil20Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        # digit = load_digits()
        path = data_path + "/coil-20-proc"
        fig_path = os.listdir(path)
        fig_path.sort()
        label = []
        data = np.zeros((1440, 128, 128))
        for i in range(1440):
            img = Image.open(path + "/" + fig_path[i])
            I_array = np.array(img)
            data[i] = I_array
            label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        data = data.reshape((data.shape[0], -1)) / 255
        print(data.shape)
        return data, np.array(label).reshape((-1))

class Coil100Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        
        path = data_path+"/coil-100"
        fig_path = os.listdir(path)

        label = []
        data = np.zeros((100 * 72, 128, 128, 3))
        for i, path_i in enumerate(fig_path):
            # print(i)
            if "obj" in path_i:
                I = Image.open(path + "/" + path_i)
                I_array = np.array(I.resize((128, 128)))
                data[i] = I_array
                label.append(int(fig_path[i].split("__")[0].split("obj")[1]))
        
        data = data.reshape((data.shape[0], -1)) / 255
        print(data.shape)
        return data, np.array(label).reshape((-1))

# class MnistSEQDataset(DigitsSEQDataset):
#     def load_data(self, data_path, train=True):
#         D = datasets.MNIST(root=data_path, train=True, download=True, transform=None)

#         data = (np.array(D.data[:60000]).astype(np.float32) / 255).reshape((60000, -1))
#         label = np.array(D.targets[:60000]).reshape((-1))
#         return data, label
    
class Coil20Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
    
        datapath = "/root/data"
        path = data_path + "/coil-20-proc"
        fig_path = os.listdir(path)
        fig_path.sort()

        label = []
        data = np.zeros((1440, 128, 128))
        for i in range(1440):
            img = Image.open(path + "/" + fig_path[i])
            I_array = np.array(img)
            data[i] = I_array
            label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        data = data.reshape((data.shape[0], -1)) / 255
        label = np.array(label)
        
        return data, label
class Cifar10VectorDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D1 = datasets.CIFAR10(root=data_path, train=True, download=True, transform=None)
        D2 = datasets.CIFAR10(root=data_path, train=False, download=True, transform=None)
        data1 = np.array(D1.data).astype(np.uint8)
        data2 = np.array(D2.data).astype(np.uint8)
        data = np.concatenate([data1,data2])
        label1 = np.array(D1.targets).reshape((-1))
        label2 = np.array(D2.targets).reshape((-1))
        label = np.concatenate([label1,label2])
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
        ])

        self.augmentation_to_tensor = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
        ])

        return data.reshape((data.shape[0], -1))/ 255, label

class Cifar100VectorDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D1 = datasets.CIFAR100(root=data_path, train=True, download=True, transform=None)
        D2 = datasets.CIFAR100(root=data_path, train=False, download=True, transform=None)
        data1 = np.array(D1.data).astype(np.uint8)
        data2 = np.array(D2.data).astype(np.uint8)
        data = np.concatenate([data1,data2])
        label1 = np.array(D1.targets).reshape((-1))
        label2 = np.array(D2.targets).reshape((-1))
        label = np.concatenate([label1,label2])
        return data.reshape((data.shape[0], -1))/ 255, label
    
class News20Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        # D1 = datasets.CIFAR100(root=data_path, train=True, download=True, transform=None)
        # D2 = datasets.CIFAR100(root=data_path, train=False, download=True, transform=None)
        # data1 = np.array(D1.data).astype(np.uint8)
        # data2 = np.array(D2.data).astype(np.uint8)
        # data = np.concatenate([data1,data2])
        # label1 = np.array(D1.targets).reshape((-1))
        # label2 = np.array(D2.targets).reshape((-1))
        # label = np.concatenate([label1,label2])

        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')
        vectorizer = TfidfVectorizer(max_features=2000, dtype=np.float32)

        x_train = torch.from_numpy(vectorizer.fit_transform(newsgroups_train.data).toarray())
        x_test = torch.from_numpy(vectorizer.transform(newsgroups_test.data).toarray())
        y_train = torch.from_numpy(newsgroups_train.target)
        y_test = torch.from_numpy(newsgroups_test.target)

        data = torch.cat([x_train, x_test], dim=0).numpy()
        labels = torch.cat([y_train, y_test], dim=0).numpy()
    
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        data = (data - mean) / std
    
    
        return data, labels
    

class OMNIDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        data_path_omni = "/zangzelin/data/difftreedata/data/"
        trainset_premerge = datasets.Omniglot(root=data_path_omni, background=True, download=False) 
        testset_premerge = datasets.Omniglot(root=data_path_omni, background=False, download=False)
        trainset_premerge_eval = datasets.Omniglot(root=data_path_omni, background=True, download=False)
        testset_premerge_eval = datasets.Omniglot(root=data_path_omni, background=False, download=False)

        # Get the corresponding labels y_train and y_test
        y_train_ind = torch.tensor([sample[1] for sample in trainset_premerge])
        y_test_ind = torch.tensor([sample[1] for sample in testset_premerge])

        # Create a list of all alphabet labels from both datasets
        alphabets = trainset_premerge._alphabets + testset_premerge._alphabets
        
        # Replace character labels by alphabet labels
        y_train_pre = []
        y_test_pre = []
        for value in y_train_ind:
            alphabet = trainset_premerge._characters[value].split("/")[0]
            alphabet_ind = alphabets.index(alphabet)
            y_train_pre.append(alphabet_ind)
        for value in y_test_ind:
            alphabet = testset_premerge._characters[value].split("/")[0]
            alphabet_ind = alphabets.index(alphabet) 
            y_test_pre.append(alphabet_ind)

        y = np.array(y_train_pre + y_test_pre)

        # Select alphabets
        num_clusters = 50
        # if num_clusters !=50:
        #     alphabets_selected = get_selected_omniglot_alphabets()[:num_clusters]
        #     alphabets_ind = []
        #     for i in alphabets_selected:
        #         alphabets_ind.append(alphabets.index(i))
        # else:
        alphabets_ind = np.arange(50)

        indx = np.array([], dtype=int)
        for i in range(num_clusters):
            indx = np.append(indx, np.where(y == alphabets_ind[i])[0])
        indx = np.sort(indx)

        # Split and stratify by digits
        digits_label = torch.concatenate([y_train_ind, y_test_ind+len(torch.unique(y_train_ind))])
        indx_train, indx_test = train_test_split(indx, test_size=0.2, random_state=0, stratify=digits_label[indx])
        indx_train = np.sort(indx_train)
        indx_test = np.sort(indx_test)

        # Define alphabets as labels
        y = y+50
        for idx, alphabet in enumerate(alphabets_ind):
            y[y==alphabet+50] = idx

        # Define mapping from digit to label
        mapping_train = []
        for value in torch.unique(y_train_ind):
            alphabet = trainset_premerge._characters[value].split("/")[0]
            alphabet_ind = alphabets.index(alphabet)
            mapping_train.append(alphabet_ind)
        mapping_test = []
        for value in torch.unique(y_test_ind):
            alphabet = testset_premerge._characters[value].split("/")[0]
            alphabet_ind = alphabets.index(alphabet)
            mapping_test.append(alphabet_ind)

        custom_target_transform_train = T.Lambda(lambda y: mapping_train[y])
        custom_target_transform_test = T.Lambda(lambda y: mapping_test[y])
        
        trainset_premerge.target_transform = custom_target_transform_train
        trainset_premerge_eval.target_transform = custom_target_transform_train
        testset_premerge.target_transform = custom_target_transform_test
        testset_premerge_eval.target_transform = custom_target_transform_test

        # Define datasets
        fullset = ConcatDataset([trainset_premerge, testset_premerge])
        fullset_eval = ConcatDataset([trainset_premerge_eval, testset_premerge_eval])
        fullset.targets = torch.from_numpy(y)
        fullset_eval.targets = torch.from_numpy(y)
        trainset = Subset(fullset, indx_train)
        trainset_eval = Subset(fullset_eval, indx_train)
        testset = Subset(fullset_eval, indx_test)

        data_list = []
        y_list = []
        for img, label in fullset:
            img28 = img.resize((28, 28))
            data_list.append(np.array(img28).flatten().astype(np.float32))  # 展平图像为一维向量
            y_list.append(label)
        
        data = np.array(data_list)
        labels = np.array(y_list)

        print("OMNI",data.shape)
    
        return data, labels