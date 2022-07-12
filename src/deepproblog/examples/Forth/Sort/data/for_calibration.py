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
  def initialize(self):
    self.connection = sqlite3.connect(Path(__file__).parent / 'raw_sort_dataset.sqlite', detect_types = sqlite3.PARSE_DECLTYPES)
    sqlite3.register_adapter(bool, int)
    sqlite3.register_converter("BOOLEAN", lambda v: bool(int(v)))
    self.cursor = self.connection.cursor()
    if not self._is_sort_samples_db_ready():
      self.cursor.execute("CREATE TABLE sort_raw_data ( X integer, Y integer, swap boolean )")
      with open(Path(__file__).parent / "swap_net.txt", "r") as swap_net:
        swap_net_samples = [ast.literal_eval(s.strip()) for s in swap_net.readlines()]
      with self.connection:
        for sample in swap_net_samples:
          X, Y = sample
          swap = self._get_swap_label(X, Y)
          self.cursor.execute("INSERT INTO sort_raw_data VALUES (:X, :Y, :swap)", {'X': X, 'Y': Y, 'swap': swap})

  def get_sample(self, i):
    self.cursor.execute(f"SELECT * FROM sort_raw_data LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result:
      return (*result,)
    else:
      return None

  def get_length(self):
    self.cursor.execute("SELECT COUNT(*) FROM sort_raw_data")
    result = self.cursor.fetchone()
    if result:
      return result[0]
    else:
      return None

  def _is_sort_samples_db_ready(self):
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'sort_raw_data';")
    sort_raw_data_table_exists = (self.cursor.fetchall() != [])
    return sort_raw_data_table_exists

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
    return F.one_hot(torch.tensor(int(label)), num_classes = 2).type(torch.FloatTensor)

def sort_dataloader_collate_fn(batch):
  inputs = []
  labels = torch.empty(0, 2)
  for e in batch:
    X, Y = e[0]
    inputs.append((X, Y))
    labels = torch.cat((labels, e[1].unsqueeze(0)), dim = 0)
  return (inputs, labels)