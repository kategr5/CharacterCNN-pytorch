import numpy as np
import pandas as pd
import string
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.datasets import fetch_20newsgroups


class AGnews(Dataset):
    def __init__(self, dataset):
        self.x_data=dataset[1]+dataset[2]
        y_data=np.array(dataset[0])
        for i in range(len(y_data)):
            y_data[i] -= 1
        self.y_data=y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        #this is where the encoding happens
        x_data=self.encoded(index)
        y_data=self.y_data[index]
        return x_data,y_data

    def encoded(self,index):
        alphabet = list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + list('’') + list('\n')
        encoded_data=torch.zeros(70,1014)
        chars=self.x_data[index]
        for index, char in enumerate(chars[::-1]):
            if char in alphabet and index<1014:
                encoded_data[alphabet.index(char)][index]=1
        return encoded_data

class Newsground(Dataset):
    def __init__(self, dataset):
        self.x_data = pd.Series(dataset.data)
        self.y_data = pd.Series(dataset.target)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        #this is where the encoding happens
        x_data=self.encoded(index)
        y_data=self.y_data[index]
        return x_data,y_data

    def encoded(self,index):
        alphabet = list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + list('’') + list('\n')
        encoded_data=torch.zeros(70,1014)
        chars=self.x_data[index]
        for index, char in enumerate(chars[::-1]):
            if char in alphabet and index<1014:
                encoded_data[alphabet.index(char)][index]=1
        return encoded_data


def load_data(dataset='<dataset>',transformation=None,n_train=None,n_test=None):


    if dataset=='20Newsground':
        train = fetch_20newsgroups(subset='train')
        test = fetch_20newsgroups(subset='test')
        trainset = Newsground(train)
        train_loader = DataLoader(dataset=trainset, batch_size=128, num_workers=0, drop_last=False)
        testset = Newsground(test)
        test_loader = DataLoader(dataset=testset, batch_size=128, num_workers=0, drop_last=False)

        return train_loader, test_loader

    else: #if dataset=='AGNews': AGNews default
        train = pd.read_csv('https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',header=None)  # 120000x3
        test = pd.read_csv('https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv',header=None)  # 7600x3
        trainset = AGnews(train)
        train_loader = DataLoader(dataset=trainset, batch_size=128, num_workers=0, drop_last=False)
        testset = AGnews(test)
        test_loader = DataLoader(dataset=testset, batch_size=128, num_workers=0, drop_last=False)

        return train_loader, test_loader