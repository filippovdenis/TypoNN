import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
from torch.utils.data import DataLoader, Dataset

class NamesDataset(Dataset):
    def __init__(self, data_path, alphabet, max_length):
        self.data_path = data_path
        self.alphabet = alphabet
        self.max_length = max_length
        self.label = []
        self.data = []
        with open(self.data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            num = 0
            for index, fields in enumerate(rdr):
                if num > 0:
                    print(fields)
                    self.data.append(fields[1])
                    self.label.append(int(fields[3]))
                num += 1
        self.y = torch.LongTensor(self.label)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        X = self.one_hot(idx)
        y = self.y[idx]
        return X, y
    def one_hot(self, idx):
        X = torch.zeros(len(self.alphabet), self.max_length)
        s = self.data[idx]
       
        for index_char, char in enumerate(s):
            if self.charIndex(char) != -1:
                X[self.charIndex(char)][index_char] = 1.0
        return X
    def charIndex(self, char):
        return self.alphabet.find(char)