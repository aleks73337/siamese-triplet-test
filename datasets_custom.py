import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

from torch.utils.data import Dataset


def get_data(data_folder: str):
    load_data = lambda x: np.load(f"{data_folder}/{x}")
    buy_patterns = load_data("buy_patterns.npy")
    buy_patterns_clusters = load_data("buy_patterns_clusters.npy")
    sell_patterns = load_data("sell_patterns.npy")
    sell_patterns_clusters = load_data("sell_patterns_clusters.npy")
    sell_patterns_clusters += 2
    patterns = np.concatenate((buy_patterns, sell_patterns), axis = 0).astype(np.float32)
    labels = np.concatenate((buy_patterns_clusters, sell_patterns_clusters), axis = 0)
    scaler = MinMaxScaler()
    for i in range(len(patterns)):
        patterns[i] = scaler.fit_transform(patterns[i])
    return patterns, labels


def split_data(patterns, labels, random_state = 50):
    x_train, x_test, y_train, y_test = train_test_split(
        patterns, labels, test_size=0.3, random_state=random_state, stratify=labels)
    return (x_train, y_train), (x_test, y_test)


class Patterns(Dataset):
    def __init__(self, root: str, train: bool, transform):
        self.train = train
        self.root = root
        self.transform = transform
        patterns, labels = get_data(root)
        train_data, test_data = split_data(patterns, labels)
        if train:
            self.patterns = torch.from_numpy(train_data[0])
            self.labels = torch.from_numpy(train_data[1])
        else:
            self.patterns = torch.from_numpy(test_data[0])
            self.labels = torch.from_numpy(test_data[1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.patterns[idx], self.labels[idx]


class SiamesePatterns(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset : Patterns):
        self.dataset = dataset

        self.train = self.dataset.train
        self.transform = self.dataset.transform

        if self.train:
            self.train_labels = self.dataset.labels
            self.train_data = self.dataset.patterns
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.dataset.labels
            self.test_data = self.dataset.patterns
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        # img1 = Image.fromarray(img1.numpy(), mode='L')
        # img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.dataset)

class TripletPatterns(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.train = self.dataset.train
        self.transform = self.dataset.transform

        if self.train:
            self.train_labels = self.dataset.labels
            self.train_data = self.dataset.patterns
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.dataset.labels
            self.test_data = self.dataset.patterns
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)