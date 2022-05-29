from abc import ABC, abstractmethod
import ast
from enum import Enum
import math
from pathlib import Path
import random
import sqlite3

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class RawSortDatasetDatabase:
  class DatasetPart(Enum):
    TRAIN = 1
    VALIDATION = 2

  def __init__(self, dataset_part = DatasetPart.VALIDATION):
    self.dataset_part = dataset_part

  def initialize(self):
    self.connection = sqlite3.connect('raw_sort_dataset.sqlite', detect_types = sqlite3.PARSE_DECLTYPES)
    sqlite3.register_adapter(bool, int)
    sqlite3.register_converter("BOOLEAN", lambda v: bool(int(v)))
    self.cursor = self.connection.cursor()
    if not self._is_sort_samples_db_ready():
      self.cursor.execute("CREATE TABLE sort_raw_data_train ( X integer, Y integer, swap boolean )")
      self.cursor.execute("CREATE TABLE sort_raw_data_validation ( X integer, Y integer, swap boolean )")
      with open(Path(__file__).parent / "swap_net.txt", "r") as swap_net:
        swap_net_samples = [ast.literal_eval(s.strip()) for s in swap_net.readlines()]
      random.shuffle(swap_net_samples)
      swap_net_split_index = math.floor(0.8 * len(swap_net_samples))
      swap_net_train = swap_net_samples[:swap_net_split_index]
      swap_net_validation = swap_net_samples[swap_net_split_index:]
      with self.connection:
        for sample in swap_net_train:
          X, Y = sample
          swap = self._get_swap_label(X, Y)
          self.cursor.execute("INSERT INTO sort_raw_data_train VALUES (:X, :Y, :swap)", {'X': X, 'Y': Y, 'swap': swap})
        for sample in swap_net_validation:
          X, Y = sample
          swap = self._get_swap_label(X, Y)
          self.cursor.execute("INSERT INTO sort_raw_data_validation VALUES (:X, :Y, :swap)", {'X': X, 'Y': Y, 'swap': swap})

  def get_sample(self, i):
    if self.dataset_part == self.__class__.DatasetPart.TRAIN:
      self.cursor.execute(f"SELECT * FROM sort_raw_data_train LIMIT 1 OFFSET {i};")
    else:
      self.cursor.execute(f"SELECT * FROM sort_raw_data_validation LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result != []:
      return (*result,)
    else:
      return None

  def get_length(self):
    if self.dataset_part == self.__class__.DatasetPart.TRAIN:
      self.cursor.execute("SELECT COUNT(*) FROM sort_raw_data_train")
    else:
      self.cursor.execute("SELECT COUNT(*) FROM sort_raw_data_validation")
    result = self.cursor.fetchone()
    if result != []:
      return result[0]
    else:
      return None

  def _is_sort_samples_db_ready(self):
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'sort_raw_data_train';")
    sort_raw_data_train_table_exists = (self.cursor.fetchall() != [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'sort_raw_data_validation';")
    sort_raw_data_validation_table_exists = (self.cursor.fetchall() != [])
    return (sort_raw_data_train_table_exists and \
            sort_raw_data_validation_table_exists)

  def _get_swap_label(self, X, Y):
    return X > Y

class RawSortValidationDataset(Dataset, ABC):
  def __init__(self):
    super(Dataset, self).__init__()
    self.dataset_db = RawSortDatasetDatabase()
    self.dataset_db.initialize()

  def __len__(self):
    return self.dataset_db.get_length()

  def __getitem__(self, idx):
    X, Y, label = self.dataset_db.get_sample(idx)
    return (X, Y), self._encode_label(label)

  def _encode_label(self, label):
    return F.one_hot(torch.tensor(label), num_classes = 2).type(torch.FloatTensor)

if __name__ == "__main__":
  ds = RawSortValidationDataset()
  print(len(ds))