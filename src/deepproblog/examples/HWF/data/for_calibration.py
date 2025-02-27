from abc import ABC, abstractmethod
import json
from pathlib import Path
from PIL import Image
import sqlite3

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms

from deepproblog.examples.HWF.data import Expression

class RawHWFDatasetDatabase:
  def initialize(self, filter):
    self.connection = sqlite3.connect(Path(__file__).parent / 'hwf_raw_dataset.sqlite')
    self.cursor = self.connection.cursor()
    if not self._is_hwf_samples_db_ready():
      self.cursor.execute("CREATE TABLE hwf_raw_data ( path text, label text, class text)")
      self.cursor.execute("CREATE TABLE hwf_raw_data_class_lengths ( class text, length integer )")
      number_samples, operator_samples = self._load_samples_from_expressions_file(filter)
      with self.connection:
        for sample, label in number_samples:
          self.cursor.execute("INSERT INTO hwf_raw_data VALUES (:path, :label, :class)", {'path': sample, 'label': label, 'class': 'number'})
        for sample, label in operator_samples:
          self.cursor.execute("INSERT INTO hwf_raw_data VALUES (:path, :label, :class)", {'path': sample, 'label': label, 'class': 'operator'})
        self.cursor.execute("INSERT INTO hwf_raw_data_class_lengths VALUES (:class, :length)", {'class': 'number', 'length': len(number_samples)})
        self.cursor.execute("INSERT INTO hwf_raw_data_class_lengths VALUES (:class, :length)", {'class': 'operator', 'length': len(operator_samples)})

  def get_numbers_sample(self, i):
    self.cursor.execute(f"SELECT * FROM hwf_raw_data WHERE class = 'number' LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result:
      return (result[0], result[1])
    else:
      return None

  def get_operators_sample(self, i):
    self.cursor.execute(f"SELECT * FROM hwf_raw_data WHERE class = 'operator' LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result:
      return (result[0], result[1])
    else:
      return None

  def get_length_numbers(self):
    self.cursor.execute("SELECT length FROM hwf_raw_data_class_lengths WHERE class = 'number'")
    result = self.cursor.fetchone()
    if result:
      return result[0]
    else:
      return None

  def get_length_operators(self):
    self.cursor.execute("SELECT length FROM hwf_raw_data_class_lengths WHERE class = 'operator'")
    result = self.cursor.fetchone()
    if result:
      return result[0]
    else:
      return None

  def _is_hwf_samples_db_ready(self):
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'hwf_raw_data';")
    hwf_raw_data_table_exists = (self.cursor.fetchall() != [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'hwf_raw_data_class_lengths';")
    hwf_raw_data_set_lengths_table_exists = (self.cursor.fetchall() != [])
    return (hwf_raw_data_table_exists and \
            hwf_raw_data_set_lengths_table_exists)

  def _is_number(self, sample):
    return ( \
      sample[1] == "0" or \
      sample[1] == "1" or \
      sample[1] == "2" or \
      sample[1] == "3" or \
      sample[1] == "4" or \
      sample[1] == "5" or \
      sample[1] == "6" or \
      sample[1] == "7" or \
      sample[1] == "8" or \
      sample[1] == "9" )

  def _is_operator(self, sample):
    return ( \
      sample[1] == "+" or \
      sample[1] == "-" or \
      sample[1] == "times" or \
      sample[1] == "div" )

  def _load_samples_from_expressions_file(self, filter):
    number_samples = []
    operator_samples = []
    
    expressions = []
    with open(Path(__file__).parent / "expr_val.json", "r") as f:
      data = json.load(f)
      for d in data:
        expression = Expression(d)
        if filter(expression.length):
          expressions.append(expression)
    for expression in expressions:
      for sample in expression.labeled_images():
        if   self._is_number(sample):
          number_samples.append(sample)
        elif self._is_operator(sample):
          operator_samples.append(sample)
    
    return number_samples, operator_samples

class RawHWFValidationDataset(Dataset, ABC):
  def __init__(self, dataset_db):
    super(Dataset, self).__init__()
    self.dataset_db = dataset_db

  @abstractmethod
  def __len__(self):
    pass

  @abstractmethod
  def __getitem__(self, idx):
    pass

class RawHWFNumbersValidationDataset(RawHWFValidationDataset):
  def __init__(self, dataset_db):
    super().__init__(dataset_db)

  def __len__(self):
    return self.dataset_db.get_length_numbers()

  def __getitem__(self, idx):
    img_path, label = self.dataset_db.get_numbers_sample(idx)
    img_path = str(Path(__file__).parent / "Handwritten_Math_Symbols" / img_path)
    image = Image.open(img_path)
    transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    image = transform(image)
    return image, self._encode_label(label)

  def _encode_label(self, number_label):
    return F.one_hot(torch.tensor(int(number_label)), num_classes = 10).type(torch.FloatTensor)

class RawHWFOperatorsValidationDataset(RawHWFValidationDataset):
  def __init__(self, dataset_db):
    super().__init__(dataset_db)

  def __len__(self):
    return self.dataset_db.get_length_operators()

  def __getitem__(self, idx):
    img_path, label = self.dataset_db.get_operators_sample(idx)
    img_path = str(Path(__file__).parent / "Handwritten_Math_Symbols" / img_path)
    image = Image.open(img_path)
    transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    image = transform(image)
    return image, self._encode_label(label)

  def _encode_label(self, operator_label):
    operator_number = (
      0 if operator_label == "+" else (
        1 if operator_label == "-" else (
          2 if operator_label == "*" else 3
        )
      )
    )
    return F.one_hot(torch.tensor(operator_number), num_classes = 4).type(torch.FloatTensor)