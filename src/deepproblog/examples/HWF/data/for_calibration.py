from abc import ABC, abstractmethod
from enum import Enum
import math
import os
from pathlib import Path
import random
import sqlite3

from torchvision.io import read_image
from torch.utils.data import Dataset

class RawHWFDatasetDatabase:
  class DatasetPart(Enum):
    TRAIN = 1
    VALIDATION = 2

  def __init__(self, dataset_part = DatasetPart.VALIDATION):
    self.dataset_part = dataset_part

  def initialize(self):
    self.connection = sqlite3.connect('hwf_raw_dataset.sqlite')
    self.cursor = self.connection.cursor()
    if not self._is_hwf_samples_db_ready():
      self.cursor.execute("CREATE TABLE hwf_raw_data_train ( path text, label text, class text)")
      self.cursor.execute("CREATE TABLE hwf_raw_data_validation ( path text, label text, class text)")
      self.cursor.execute("CREATE TABLE hwf_raw_data_class_lengths_train ( class text, length integer )")
      self.cursor.execute("CREATE TABLE hwf_raw_data_class_lengths_validation ( class text, length integer )")
      image_root = Path(__file__).parent / "Handwritten_Math_Symbols"
      number_samples = [(sample, data_dir.name) for data_dir in image_root.iterdir() if data_dir.is_dir() and self._is_number_dir(data_dir) for sample in data_dir.iterdir()]
      operator_samples = [(sample, data_dir.name) for data_dir in image_root.iterdir() if data_dir.is_dir() and self._is_operator_dir(data_dir) for sample in data_dir.iterdir()]
      random.shuffle(number_samples)
      random.shuffle(operator_samples)
      number_split_index = math.floor(0.8 * len(number_samples))
      operator_split_index = math.floor(0.8 * len(operator_samples))
      number_samples_train = number_samples[:number_split_index]
      number_samples_validation = number_samples[number_split_index:]
      operator_samples_train = operator_samples[:operator_split_index]
      operator_samples_validation = operator_samples[operator_split_index:]
      with self.connection:
        for sample, label in number_samples_train:
          self.cursor.execute("INSERT INTO hwf_raw_data_train VALUES (:path, :label, :class)", {'path': str(sample.resolve()), 'label': label, 'class': 'number'})
        for sample, label in number_samples_validation:
          self.cursor.execute("INSERT INTO hwf_raw_data_validation VALUES (:path, :label, :class)", {'path': str(sample.resolve()), 'label': label, 'class': 'number'})
        for sample, label in operator_samples_train:
          self.cursor.execute("INSERT INTO hwf_raw_data_train VALUES (:path, :label, :class)", {'path': str(sample.resolve()), 'label': label, 'class': 'operator'})
        for sample, label in operator_samples_validation:
          self.cursor.execute("INSERT INTO hwf_raw_data_validation VALUES (:path, :label, :class)", {'path': str(sample.resolve()), 'label': label, 'class': 'operator'})
        self.cursor.execute("INSERT INTO hwf_raw_data_class_lengths_train VALUES (:class, :length)", {'class': 'number', 'length': len(number_samples_train)})
        self.cursor.execute("INSERT INTO hwf_raw_data_class_lengths_train VALUES (:class, :length)", {'class': 'operator', 'length': len(operator_samples_train)})
        self.cursor.execute("INSERT INTO hwf_raw_data_class_lengths_validation VALUES (:class, :length)", {'class': 'number', 'length': len(number_samples_validation)})
        self.cursor.execute("INSERT INTO hwf_raw_data_class_lengths_validation VALUES (:class, :length)", {'class': 'operator', 'length': len(operator_samples_validation)})

  def get_numbers_sample(self, i):
    if self.dataset_part == self.__class__.DatasetPart.TRAIN:
      self.cursor.execute(f"SELECT * FROM hwf_raw_data_train WHERE class = 'number' LIMIT 1 OFFSET {i};")
    else:
      self.cursor.execute(f"SELECT * FROM hwf_raw_data_validation WHERE class = 'number' LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result != []:
      return (result[0], result[1])
    else:
      return None

  def get_operators_sample(self, i):
    if self.dataset_part == self.__class__.DatasetPart.TRAIN:
      self.cursor.execute(f"SELECT * FROM hwf_raw_data_train WHERE class = 'operator' LIMIT 1 OFFSET {i};")
    else:
      self.cursor.execute(f"SELECT * FROM hwf_raw_data_validation WHERE class = 'operator' LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result != []:
      return (result[0], result[1])
    else:
      return None

  def get_length_numbers(self):
    if self.dataset_part == self.__class__.DatasetPart.TRAIN:
      self.cursor.execute("SELECT length FROM hwf_raw_data_class_lengths_train WHERE class = 'number'")
    else:
      self.cursor.execute("SELECT length FROM hwf_raw_data_class_lengths_validation WHERE class = 'number'")
    result = self.cursor.fetchone()
    if result != []:
      return result[0]
    else:
      return None

  def get_length_operators(self):
    if self.dataset_part == self.__class__.DatasetPart.TRAIN:
      self.cursor.execute("SELECT length FROM hwf_raw_data_class_lengths_train WHERE class = 'operator'")
    else:
      self.cursor.execute("SELECT length FROM hwf_raw_data_class_lengths_validation WHERE class = 'operator'")
    result = self.cursor.fetchone()
    if result != []:
      return result[0]
    else:
      return None

  def _is_hwf_samples_db_ready(self):
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'hwf_raw_data_train';")
    hwf_raw_data_train_table_exists = (self.cursor.fetchall() != [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'hwf_raw_data_class_lengths_train';")
    hwf_raw_data_set_lengths_train_table_exists = (self.cursor.fetchall() != [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'hwf_raw_data_validation';")
    hwf_raw_data_validation_table_exists = (self.cursor.fetchall() != [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'hwf_raw_data_class_lengths_validation';")
    hwf_raw_data_set_lengths_validation_table_exists = (self.cursor.fetchall() != [])
    return (hwf_raw_data_train_table_exists and \
            hwf_raw_data_set_lengths_train_table_exists and \
            hwf_raw_data_validation_table_exists and \
            hwf_raw_data_set_lengths_validation_table_exists)

  def _is_number_dir(self, directory):
    return ( \
      directory.name == "0" or \
      directory.name == "1" or \
      directory.name == "2" or \
      directory.name == "3" or \
      directory.name == "4" or \
      directory.name == "5" or \
      directory.name == "6" or \
      directory.name == "7" or \
      directory.name == "8" or \
      directory.name == "9" )

  def _is_operator_dir(self, directory):
    return ( \
      directory.name == "+" or \
      directory.name == "-" or \
      directory.name == "times" or \
      directory.name == "div" )

class RawHWFValidationDataset(Dataset, ABC):
  def __init__(self):
    super(Dataset, self).__init__()
    self.dataset_db = RawHWFDatasetDatabase()
    self.dataset_db.initialize()

  @abstractmethod
  def __len__(self):
    pass

  @abstractmethod
  def __getitem__(self, idx):
    pass

    return self.dataset_db.get_length_numbers()

class RawHWFNumbersValidationDataset(RawHWFValidationDataset):
  def __init__(self):
    super().__init__()

  def __len__(self):
    return self.dataset_db.get_length_numbers()

  def __getitem__(self, idx):
    img_path, label = self.dataset_db.get_numbers_sample(idx)
    image = read_image(img_path)
    return image, label

class RawHWFOperatorsValidationDataset(RawHWFValidationDataset):
  def __init__(self):
    super().__init__()

  def __len__(self):
    return self.dataset_db.get_length_operators()

  def __getitem__(self, idx):
    img_path, label = self.dataset_db.get_operators_sample(idx)
    image = read_image(img_path)
    return image, label