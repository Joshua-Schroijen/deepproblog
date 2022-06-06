import os

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from deepproblog.dataset import ImageDataset, Subset
from deepproblog.query import Query
from problog.logic import Term, Constant

path = os.path.dirname(os.path.abspath(__file__))

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

class Coins(ImageDataset):
    def __init__(
        self,
        set_part,
    ):
        super().__init__("{}/image_data/{}/".format(path, set_part), transform=transform)
        self.data = []
        self.set_part = set_part
        with open("{}/label_data/{}.csv".format(path, set_part)) as f:
            for line in f:
                c1, c2 = [l.strip() for l in line.split(",")]
                outcome = "loss"
                if c1 == c2:
                    outcome = "win"
                self.data.append((c1, c2, outcome))

    def to_query(self, i):
        c1, c2, outcome = self.data[i]
        sub = {Term("a"): Term("tensor", Term(self.set_part, Constant(i)))}
        # if j == 0:
        #     return Term('coin', Constant(j + 1), Term('a'), Term(c1)), sub
        # elif j == 1:
        #     return Term('coin', Constant(j + 1), Term('a'), Term(c2)), sub
        # else:
        return Query(Term("game", Term("a"), Term(outcome)), sub)

    def __len__(self):
        return len(self.data)

class RawCoinsValidationDataset(TorchDataset):
    def __init__(self, coins_dataset):
        self.coins_dataset = coins_dataset
        self.labels = self._get_labels()

    def _get_labels(self):
        labels = []
        line_no = 0
        coins_dataset_is_subset = isinstance(self.coins_dataset, Subset)
        with open("{}/label_data/{}.csv".format(path, self.set_part)) as f:
            for line in f:
                if coins_dataset_is_subset and not \
                   (line_no < self.coins_dataset.j and \
                    line_no >= self.coins_dataset.i):
                    continue
                else:
                    c1, c2 = [l.strip() for l in line.split(",")]
                    labels.append((c1, c2))

                line_no += 1

    def __len__(self):
        return len(self.coins_dataset)

    def _encode_coin_label(coin_label):
        return F.one_hot(torch.tensor((0 if coin_label == "heads" else 1)), num_classes = 2).type(torch.FloatTensor)

class RawCoinsNet1ValidationDataset(RawCoinsValidationDataset):
    def __init__(
        self,
        coins_dataset,
    ):
        super().__init__(coins_dataset)

    def __getitem__(self, idx):
        return self.coins_dataset[idx], self._encode_coin_label(self.labels[idx][0])

class RawCoinsNet2ValidationDataset(RawCoinsValidationDataset):
    def __init__(
        self,
        coins_dataset,
    ):
        super().__init__(coins_dataset)

    def __getitem__(self, idx):
        return self.coins_dataset[idx], self._encode_coin_label(self.labels[idx][1])

train_dataset = Coins("train")
test_dataset = Coins("test")
