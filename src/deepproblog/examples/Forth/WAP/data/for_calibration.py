from abc import ABC, abstractmethod
import ast
import csv
from enum import Enum
import math
from pathlib import Path
import random
import re
import sqlite3

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import problog.logic

from deepproblog.utils import bytes_to_tensor, tensor_to_bytes

class RawWAPDatasetDatabase:
  def initialize(self):
    self.connection = sqlite3.connect(Path(__file__).parent / 'raw_wap_dataset.sqlite', detect_types = sqlite3.PARSE_DECLTYPES)
    sqlite3.register_adapter(bool, int)
    sqlite3.register_converter("BOOLEAN", lambda v: bool(int(v)))
    self.cursor = self.connection.cursor()
    if not self._is_WAP_samples_db_ready():
      self.cursor.execute("CREATE TABLE wap_op1_raw_data ( WAP text, embedding blob, operator integer )")
      self.cursor.execute("CREATE TABLE wap_op2_raw_data ( WAP text, embedding blob, operator integer )")
      self.cursor.execute("CREATE TABLE wap_permute_raw_data ( WAP text, embedding blob, permutation integer )")
      self.cursor.execute("CREATE TABLE wap_swap_raw_data ( WAP text, embedding blob, swap boolean )")
      samples = self._read_samples_from_file()
      with self.connection:
        for sample in samples:
          zeros_embedding = tensor_to_bytes(torch.zeros(4096))
          WAP, op1, op2, permutation, swapped = sample
          self.cursor.execute("INSERT INTO wap_op1_raw_data VALUES (:WAP, :embedding, :operator)", {'WAP': WAP, 'embedding': zeros_embedding, 'operator': op1})
          self.cursor.execute("INSERT INTO wap_op2_raw_data VALUES (:WAP, :embedding, :operator)", {'WAP': WAP, 'embedding': zeros_embedding, 'operator': op2})
          self.cursor.execute("INSERT INTO wap_permute_raw_data VALUES (:WAP, :embedding, :permutation)", {'WAP': WAP, 'embedding': zeros_embedding, 'permutation': permutation})
          self.cursor.execute("INSERT INTO wap_swap_raw_data VALUES (:WAP, :embedding, :swap)", {'WAP': WAP, 'embedding': zeros_embedding, 'swap': swapped})

  def get_sample_op1(self, i):
    self.cursor.execute(f"SELECT * FROM wap_op1_raw_data LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result:
      return (*result,)
    else:
      return None

  def get_sample_op2(self, i):
    self.cursor.execute(f"SELECT * FROM wap_op2_raw_data LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result:
      return (*result,)
    else:
      return None

  def get_sample_permute(self, i):
    self.cursor.execute(f"SELECT * FROM wap_permute_raw_data LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result:
      return (*result,)
    else:
      return None

  def get_sample_swap(self, i):
    self.cursor.execute(f"SELECT * FROM wap_swap_raw_data LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result:
      return (*result,)
    else:
      return None

  def get_length_op1(self):
    self.cursor.execute("SELECT COUNT(*) FROM wap_op1_raw_data")
    result = self.cursor.fetchone()
    if result:
      return result[0]
    else:
      return None

  def get_length_op2(self):
    self.cursor.execute("SELECT COUNT(*) FROM wap_op2_raw_data")
    result = self.cursor.fetchone()
    if result:
      return result[0]
    else:
      return None

  def get_length_permute(self):
    self.cursor.execute("SELECT COUNT(*) FROM wap_permute_raw_data")
    result = self.cursor.fetchone()
    if result:
      return result[0]
    else:
      return None

  def get_length_swap(self):
    self.cursor.execute("SELECT COUNT(*) FROM wap_swap_raw_data")
    result = self.cursor.fetchone()
    if result:
      return result[0]
    else:
      return None

  def update_embedding_op1(self, WAP, rnn):
    new_embedding = tensor_to_bytes(rnn.forward(problog.logic.Constant(WAP)))
    with self.connection:
      self.cursor.execute("UPDATE wap_op1_raw_data SET embedding = ? WHERE WAP = ?;", [new_embedding, WAP]) 

  def update_embeddings_op1(self, rnn):
    self.cursor.execute(f"SELECT WAP FROM wap_op1_raw_data;")
    results = self.cursor.fetchall()
    for result in results:
      WAP, = result
      self.update_embedding_op1(WAP, rnn)

  def update_embedding_op2(self, WAP, rnn):
    new_embedding = tensor_to_bytes(rnn.forward(problog.logic.Constant(WAP)))
    with self.connection:
      self.cursor.execute("UPDATE wap_op2_raw_data SET embedding = ? WHERE WAP = ?;", [new_embedding, WAP]) 

  def update_embeddings_op2(self, rnn):
    self.cursor.execute(f"SELECT WAP FROM wap_op2_raw_data;")
    results = self.cursor.fetchall()
    for result in results:
      WAP, = result
      self.update_embedding_op2(WAP, rnn)

  def update_embedding_permute(self, WAP, rnn):
    new_embedding = tensor_to_bytes(rnn.forward(problog.logic.Constant(WAP)))
    with self.connection:
      self.cursor.execute("UPDATE wap_permute_raw_data SET embedding = ? WHERE WAP = ?;", [new_embedding, WAP]) 

  def update_embeddings_permute(self, rnn):
    self.cursor.execute(f"SELECT WAP FROM wap_permute_raw_data;")
    results = self.cursor.fetchall()
    for result in results:
      WAP, = result
      self.update_embedding_permute(WAP, rnn)

  def update_embedding_swap(self, WAP, rnn):
    new_embedding = tensor_to_bytes(rnn.forward(problog.logic.Constant(WAP)))
    with self.connection:
      self.cursor.execute("UPDATE wap_swap_raw_data SET embedding = ? WHERE WAP = ?;", [new_embedding, WAP]) 

  def update_embeddings_swap(self, rnn):
    self.cursor.execute(f"SELECT WAP FROM wap_swap_raw_data;")
    results = self.cursor.fetchall()
    for result in results:
      WAP, = result
      self.update_embedding_swap(WAP, rnn)

  def _is_WAP_samples_db_ready(self):
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'wap_op1_raw_data';")
    wap_op1_raw_data_table_exists = (self.cursor.fetchall() != [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'wap_op2_raw_data';")
    wap_op2_raw_data_table_exists = (self.cursor.fetchall() != [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'wap_permute_raw_data';")
    wap_permute_raw_data_table_exists = (self.cursor.fetchall() != [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'wap_swap_raw_data';")
    wap_swap_raw_data_table_exists = (self.cursor.fetchall() != [])
    return (wap_op1_raw_data_table_exists and \
            wap_op2_raw_data_table_exists and \
            wap_permute_raw_data_table_exists and
            wap_swap_raw_data_table_exists)

  def _read_samples_from_file(self):
    inputs = []
    with open(Path(__file__).parent / 'dev_formulas.csv', 'r') as dev_formulas_file:
      dev_formulas_rows = list(csv.reader(dev_formulas_file, delimiter = ','))
    for dev_formulas_row in dev_formulas_rows:
      WAP, formula, _ = dev_formulas_row
      formula = formula.strip()
      operators = re.findall("([\+\-\*/])", formula)
      numbers = self._get_numbers(WAP)
      numbers_permuted = self._get_numbers(formula)
      permutation = self._get_permutation(numbers, numbers_permuted)
      swapped = self._get_swapped(formula)
      inputs.append([WAP, *operators, permutation, swapped])

    return inputs

  def _get_numbers(self, s):
    return [int(r) for r in re.findall(r"\b(\d+)\b", s)]

  def _get_permutation(self, numbers, permutation):
    if   numbers[0] == permutation[0] and \
         numbers[1] == permutation[1]: 
      return 0
    elif numbers[0] == permutation[0] and \
         numbers[1] == permutation[2]:
      return 1
    elif numbers[0] == permutation[1] and \
         numbers[1] == permutation[0]:
      return 2
    elif numbers[0] == permutation[2] and \
         numbers[1] == permutation[0]:
      return 3
    elif numbers[0] == permutation[1] and \
         numbers[1] == permutation[2]:
      return 4
    elif numbers[0] == permutation[2] and \
         numbers[1] == permutation[1]:
      return 5

  def _get_swapped(self, formula):
    return re.match(".*[\+\-\*/]\(.*", formula) != None

class RawWAPValidationDataset(Dataset, ABC):
  def __init__(self):
    super(Dataset, self).__init__()
    self.dataset_db = RawWAPDatasetDatabase()
    self.dataset_db.initialize()

  @abstractmethod
  def __len__(self):
    pass

  @abstractmethod
  def __getitem__(self, idx):
    pass

  @abstractmethod
  def update_embedding(self, WAP, rnn):
    pass

  @abstractmethod
  def update_embeddings(self, rnn):
    pass

class RawWAPOpValidationDataset(RawWAPValidationDataset):
  mapping = {
    "+": 0,
    "-": 1,
    "*": 2,
    "/": 3
  }

  def __init__(self):
    super().__init__()

  def _encode_operator(self, operator):
    return F.one_hot(torch.tensor(self.mapping[operator]), num_classes = 4).type(torch.FloatTensor)

class RawWAPOp1ValidationDataset(RawWAPOpValidationDataset):
  def __init__(self):
    super().__init__()

  def __len__(self):
    return self.dataset_db.get_length_op1()

  def __getitem__(self, idx):
    _, embedding, operator = self.dataset_db.get_sample_op1(idx)
    return bytes_to_tensor(embedding), self._encode_operator(operator)

  def update_embedding(self, WAP, rnn):
    self.dataset_db.update_embedding_op1(WAP, rnn)

  def update_embeddings(self, rnn):
    self.dataset_db.update_embeddings_op1(rnn)

class RawWAPOp2ValidationDataset(RawWAPOpValidationDataset):
  def __init__(self):
    super().__init__()

  def __len__(self):
    return self.dataset_db.get_length_op2()

  def __getitem__(self, idx):
    _, embedding, operator = self.dataset_db.get_sample_op2(idx)
    return bytes_to_tensor(embedding), self._encode_operator(operator)

  def update_embedding(self, WAP, rnn):
    self.dataset_db.update_embedding_op2(WAP, rnn)

  def update_embeddings(self, rnn):
    self.dataset_db.update_embeddings_op2(rnn)

class RawWAPPermuteValidationDataset(RawWAPValidationDataset):
  def __init__(self):
    super().__init__()

  def __len__(self):
    return self.dataset_db.get_length_permute()

  def __getitem__(self, idx):
    _, embedding, permutation = self.dataset_db.get_sample_permute(idx)
    return bytes_to_tensor(embedding), self._encode_permutation(permutation)

  def update_embedding(self, WAP, rnn):
    self.dataset_db.update_embedding_permute(WAP, rnn)

  def update_embeddings(self, rnn):
    self.dataset_db.update_embeddings_permute(rnn)

  def _encode_permutation(self, permutation_number):
    return F.one_hot(torch.tensor(permutation_number), num_classes = 6).type(torch.FloatTensor)

class RawWAPSwapValidationDataset(RawWAPValidationDataset):
  def __init__(self):
    super().__init__()

  def __len__(self):
    return self.dataset_db.get_length_swap()

  def __getitem__(self, idx):
    _, embedding, swap = self.dataset_db.get_sample_swap(idx)
    return bytes_to_tensor(embedding), self._encode_swap(swap)

  def update_embedding(self, WAP, rnn):
    self.dataset_db.update_embedding_swap(WAP, rnn)

  def update_embeddings(self, rnn):
    self.dataset_db.update_embeddings_swap(rnn)

  def _encode_swap(self, swapped):
    return F.one_hot(torch.tensor(int(swapped)), num_classes = 2).type(torch.FloatTensor)