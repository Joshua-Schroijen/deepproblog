from abc import ABC, abstractmethod
import ast
from enum import Enum
import math
import os
from pathlib import Path
import random
import sqlite3

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

class RawAddDatasetDatabase:
  class DatasetPart(Enum):
    TRAIN = 1
    VALIDATION = 2

  def __init__(self, dataset_part = DatasetPart.VALIDATION):
    self.dataset_part = dataset_part

  def initialize(self):
    self.__class__.DatasetPart
    self.connection = sqlite3.connect('raw_add_dataset.sqlite')
    self.cursor = self.connection.cursor()
    if self._is_add_samples_db_ready():
      self.cursor.execute("CREATE TABLE add_neural1_raw_data_train ( I1 integer, I2 integer, Carry integer, O integer)")
      self.cursor.execute("CREATE TABLE add_neural2_raw_data_train ( I1 integer, I2 integer, Carry integer, NewCarry integer)")
      self.cursor.execute("CREATE TABLE add_neural1_raw_data_validation ( I1 integer, I2 integer, Carry integer, O integer)")
      self.cursor.execute("CREATE TABLE add_neural2_raw_data_validation ( I1 integer, I2 integer, Carry integer, NewCarry integer)")
      self.cursor.execute("CREATE TABLE add_raw_data_lengths_train ( predicate text, length integer )")
      self.cursor.execute("CREATE TABLE add_raw_data_lengths_validation ( predicate text, length integer )")
      with open(Path(__file__).parent / "neural1.txt", "r") as neural1_file:
        neural1 = [ast.literal_eval(s.strip()) for s in neural1_file.readlines()]
      with open(Path(__file__).parent / "neural2.txt", "r") as neural2_file:
        neural2 = [ast.literal_eval(s.strip()) for s in neural2_file.readlines()]
      random.shuffle(neural1)
      random.shuffle(neural2)
      neural1_split_index = math.floor(0.8 * len(neural1))
      neural2_split_index = math.floor(0.8 * len(neural2))
      neural1_train = neural1[:neural1_split_index]
      neural1_validation = neural1[neural1_split_index:]
      neural2_train = neural2[:neural2_split_index]
      neural2_validation = neural2[neural2_split_index:]
      with self.connection:
        for sample in neural1_train:
          I1, I2, Carry = sample
          O = self._get_neural1_label(I1, I2, Carry)
          self.cursor.execute("INSERT INTO add_neural1_raw_data_train VALUES (:I1, :I2, :Carry, :O)", {'I1': I1, 'I2': I2, 'Carry': Carry, 'O': 0})
        for sample in neural1_validation:
          I1, I2, Carry = sample
          O = self._get_neural1_label(I1, I2, Carry)
          self.cursor.execute("INSERT INTO add_neural1_raw_data_validation VALUES (:I1, :I2, :Carry, :O)", {'I1': I1, 'I2': I2, 'Carry': Carry, 'O': 0})
        for sample in neural2_train:
          I1, I2, Carry = sample
          NewCarry = self._get_neural2_label(I1, I2, Carry)
          self.cursor.execute("INSERT INTO add_neural2_raw_data_train VALUES (:I1, :I2, :Carry, :NewCarry)", {'I1': I1, 'I2': I2, 'Carry': Carry, 'NewCarry': NewCarry})
        for sample in neural2_validation:
          I1, I2, Carry = sample
          NewCarry = self._get_neural1_label(I1, I2, Carry)
          self.cursor.execute("INSERT INTO add_neural2_raw_data_validation VALUES (:I1, :I2, :Carry, :NewCarry)", {'I1': I1, 'I2': I2, 'Carry': Carry, 'NewCarry': NewCarry})
        self.cursor.execute("INSERT INTO add_raw_data_lengths_train VALUES (:predicate, :length)", {'predicate': 'neural1', 'length': len(neural1_train)})
        self.cursor.execute("INSERT INTO add_raw_data_lengths_train VALUES (:predicate, :length)", {'predicate': 'neural2', 'length': len(neural2_train)})
        self.cursor.execute("INSERT INTO add_raw_data_lengths_validation VALUES (:predicate, :length)", {'predicate': 'neural1', 'length': len(neural1_validation)})
        self.cursor.execute("INSERT INTO add_raw_data_lengths_validation VALUES (:predicate, :length)", {'predicate': 'neural2', 'length': len(neural2_validation)})

  def get_neural1_sample(self, i):
    if self.dataset_part == self.__class__.DatasetPart.TRAIN:
      self.cursor.execute(f"SELECT * FROM add_neural1_raw_data_train LIMIT 1 OFFSET {i};")
    else:
      self.cursor.execute(f"SELECT * FROM add_neural1_raw_data_validation LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result != []:
      return (*result,)
    else:
      return None

  def get_neural2_sample(self, i):
    if self.dataset_part == self.__class__.DatasetPart.TRAIN:
      self.cursor.execute(f"SELECT * FROM add_neural2_raw_data_train LIMIT 1 OFFSET {i};")
    else:
      self.cursor.execute(f"SELECT * FROM add_neural2_raw_data_validation LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result != []:
      return (*result,)
    else:
      return None

  def get_length_neural1(self):
    if self.dataset_part == self.__class__.DatasetPart.TRAIN:
      self.cursor.execute("SELECT length FROM add_raw_data_lengths_train WHERE predicate = 'neural1'")
    else:
      self.cursor.execute("SELECT length FROM add_raw_data_lengths_validation WHERE predicate = 'neural1'")
    result = self.cursor.fetchone()
    if result != []:
      return result[0]
    else:
      return None

  def get_length_neural2(self):
    if self.dataset_part == self.__class__.DatasetPart.TRAIN:
      self.cursor.execute("SELECT length FROM add_raw_data_lengths_train WHERE predicate = 'neural2'")
    else:
      self.cursor.execute("SELECT length FROM add_raw_data_lengths_validation WHERE predicate = 'neural2'")
    result = self.cursor.fetchone()
    if result != []:
      return result[0]
    else:
      return None

  def _is_add_samples_db_ready(self):
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'add_neural1_raw_data_train';")
    add_neural1_raw_data_train_table_exists = (self.cursor.fetchall() == [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'add_neural2_raw_data_train';")
    add_neural2_raw_data_train_table_exists = (self.cursor.fetchall() == [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'add_neural1_raw_data_validation';")
    add_neural1_raw_data_validation_table_exists = (self.cursor.fetchall() == [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'add_neural2_raw_data_validation';")
    add_neural2_raw_data_validation_table_exists = (self.cursor.fetchall() == [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'add_raw_data_lengths_train';")
    add_raw_data_lengths_train_table_exists = (self.cursor.fetchall() == [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'add_raw_data_lengths_validation';")
    add_raw_data_lengths_validation_table_exists = (self.cursor.fetchall() == [])
    return (add_neural1_raw_data_train_table_exists and \
            add_neural2_raw_data_train_table_exists and \
            add_neural1_raw_data_validation_table_exists and \
            add_neural2_raw_data_validation_table_exists and \
            add_raw_data_lengths_train_table_exists and \
            add_raw_data_lengths_validation_table_exists)

  def _get_neural1_label(self, I1, I2, Carry):
    return ((I1 + I2 + Carry) % 10)

  def _get_neural2_label(self, I1, I2, Carry):
    return ((I1 + I2 + Carry) // 10)

class RawAddValidationDataset(Dataset, ABC):
  def __init__(self):
    super(Dataset, self).__init__()
    self.dataset_db = RawAddDatasetDatabase()
    self.dataset_db.initialize()

  @abstractmethod
  def __len__(self):
    pass

  @abstractmethod
  def __getitem__(self, idx):
    pass

class RawAddNeural1ValidationDataset(RawAddValidationDataset):
  def __init__(self):
    super().__init__()

  def __len__(self):
    return self.dataset_db.get_length_neural1()

  def __getitem__(self, idx):
    I1, I2, Carry, O = self.dataset_db.get_neural1_sample(idx)
    return (I1, I2, Carry), self._encode_label(O)

  def _encode_label(self, label):
    return F.one_hot(torch.tensor(label), num_classes = 10).type(torch.FloatTensor)


class RawAddNeural2ValidationDataset(RawAddValidationDataset):
  def __init__(self):
    super().__init__()

  def __len__(self):
    return self.dataset_db.get_length_neural2()

  def __getitem__(self, idx):
    I1, I2, Carry, NewCarry = self.dataset_db.get_neural2_sample(idx)
    return (I1, I2, Carry), NewCarry

  def _encode_label(self, label):
    return F.one_hot(torch.tensor(label), num_classes = 2).type(torch.FloatTensor)

if __name__ == "__main__":
  neural1_validation_dataset = RawAddNeural1ValidationDataset()
  print(len(neural1_validation_dataset))