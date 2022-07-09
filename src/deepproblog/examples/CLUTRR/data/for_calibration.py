from abc import ABC, abstractmethod
import ast
from pathlib import Path
import sqlite3

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from problog.logic import list2term, Term, Constant

from deepproblog.utils import bytes_to_tensor, tensor_to_bytes, parse, term2list2

class RawCLUTRRDatasetDatabase:
  def initialize(self):
    self.connection = sqlite3.connect(Path(__file__).parent / 'raw_CLUTRR_dataset.sqlite')
    self.cursor = self.connection.cursor()
    if not self._is_CLUTRR_samples_db_ready():
      self.cursor.execute("CREATE TABLE CLUTRR_rel_extract_raw_data ( sample_id integer, sentence text, entity_1 integer, entity_2 integer, embedding_part_1 blob, embedding_part_2 blob, relation text )")
      self.cursor.execute("CREATE TABLE CLUTRR_gender_net_raw_data ( sentences text, entity integer, gender integer )")
      with self.connection:
        zeros_embedding = tensor_to_bytes(torch.zeros(4096))
        for sample in self._get_rel_extract_samples():
          sample_id, sentence, entity_1, entity_2, relation = sample
          self.cursor.execute("SELECT * FROM CLUTRR_rel_extract_raw_data WHERE sample_id = ? AND sentence = ? AND entity_1 = ? AND entity_2 = ?;", [sample_id, sentence, entity_1, entity_2])
          if self.cursor.fetchone() == []:
            self.cursor.execute("INSERT INTO CLUTRR_rel_extract_raw_data VALUES (:sample_id, :sentence, :entity_1, :entity_2)", {'sample_id': sample_id, 'sentence': sentence, 'entity_1': entity_1, 'entity_2': entity_2, 'embedding_part_1': zeros_embedding, 'embedding_part_2': zeros_embedding, 'relation': relation})
        for sample in self._get_gender_net_samples():
          sentences, entity, gender = sample
          self.cursor.execute("INSERT INTO CLUTRR_gender_net_raw_data VALUES (:sentences, :entity, :gender)", {'sentences': sentences, 'entity': entity, 'gender': gender})

  def get_gender_net_sample(self, i):
    self.cursor.execute(f"SELECT * FROM CLUTRR_gender_net_raw_data LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result != []:
      return (*result,)
    else:
      return None

  def get_rel_extract_sample(self, i):
    self.cursor.execute(f"SELECT * FROM CLUTRR_rel_extract_raw_data LIMIT 1 OFFSET {i};")
    result = self.cursor.fetchone()
    if result != []:
      return (*result,)
    else:
      return None

  def get_length_rel_extract(self):
    self.cursor.execute("SELECT COUNT(*) FROM CLUTRR_rel_extract_raw_data")
    result = self.cursor.fetchone()
    if result != []:
      return result[0]
    else:
      return None

  def get_length_gender_net(self):
    self.cursor.execute("SELECT COUNT(*) FROM CLUTRR_gender_net_raw_data")
    result = self.cursor.fetchone()
    if result != []:
      return result[0]
    else:
      return None

  def update_embedding_rel_extract(self, sentence, entity_1, entity_2, encoder):
    new_embeddings = [tensor_to_bytes(output) for output in encoder.forward(Constant(sentence), Constant(entity_1), Constant(entity_2))]
    with self.connection:
      self.cursor.execute("UPDATE CLUTRR_rel_extract_raw_data SET embedding_part_1 = ? embedding_part_2 = ? WHERE sentence = ? AND entity_1 = ? AND entity_2 = ?;", [new_embeddings[0], new_embeddings[1], sentence, entity_1, entity_2]) 

  def update_embeddings_rel_extract(self, encoder):
    self.cursor.execute(f"SELECT sentence, entity_1, entity_2 FROM CLUTRR_rel_extract_raw_data;")
    results = self.cursor.fetchall()
    for result in results:
      sentence, entity_1, entity_2 = result
      self.update_embedding_rel_extract(sentence, entity_1, entity_2, encoder)

  def _is_CLUTRR_samples_db_ready(self):
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'CLUTRR_rel_extract_raw_data';")
    CLUTRR_rel_extract_raw_data_table_exists = (self.cursor.fetchall() != [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'CLUTRR_gender_net_raw_data';")
    CLUTRR_gender_net_raw_data_table_exists = (self.cursor.fetchall() != [])
    return (CLUTRR_rel_extract_raw_data_table_exists and \
            CLUTRR_gender_net_raw_data_table_exists)

  def _get_gender_net_samples(self):
    convert = lambda x: x.strip()

    gender_net_samples = []
    with open(Path(__file__).parent / "gender_net_calibration_validation.txt", "r") as gender_net_file:
      done = False
      while not done:
        try:
          sample_genders_dict = ast.literal_eval(convert(next(gender_net_file)))
          sample_network_inputs = term2list2(
            parse(
              convert(next(gender_net_file)) + "."
            )
          )
          sentences = str(list2term(sample_network_inputs[:-1]))
          entity = int(sample_network_inputs[-1])
          gender = sample_genders_dict[entity]
          gender_net_samples.appen((sentences, entity, gender))
        except StopIteration:
          done = True
    return gender_net_samples

  def _get_rel_extract_samples(self):
    convert = lambda x: x.decode('ascii').strip()

    rel_extract_samples = []
    sample_locations = [(0, -1)]
    with open(Path(__file__).parent / "rel_extract_calibration_validation.txt", "rb") as rel_extract_file:
      offset = 0
      no_lines = 0
      prev_line_begins_with_brace = True
      for line in rel_extract_file:
        line_length = len(line)
        line = convert(line)
        if line[0] == '{' and \
           not prev_line_begins_with_brace:
          sample_locations[-1] = (sample_locations[-1][0], no_lines) 
          sample_locations.append((offset, -1))
          no_lines = 0
        if line[0] == '{':
          prev_line_begins_with_brace = True
        else:
          prev_line_begins_with_brace = False
        offset += line_length
        no_lines += 1
      sample_locations[-1] = (sample_locations[-1][0], no_lines)
      rel_extract_file.seek(0)
      for sample_id, sample_location in enumerate(sample_locations):
        sample_offset, sample_lines = sample_location
        rel_extract_file.seek(sample_offset)
        sample_relations_dict = ast.literal_eval(convert(next(rel_extract_file)))
        next(rel_extract_file)
        self._extend_relations_dict(sample_relations_dict)
        for _ in range(sample_lines - 2):
          sentence, entity_1, entity_2 = convert(next(rel_extract_file)).split(",")
          entity_1 = int(entity_1)
          entity_2 = int(entity_2)
          relation = sample_relations_dict[(entity_1, entity_2)]
        rel_extract_samples.append((sample_id, sentence, entity_1, entity_2, relation))
    return rel_extract_samples

  def _extend_relations_dict(self, relations_dict):
    relations_to_create = []
    for existing_relation in relations_dict.keys():
      reversed_relation = (existing_relation[1], existing_relation[0])
      if not reversed_relation in relations_dict:
        relations_to_create.append((existing_relation, reversed_relation))

    for relation_to_create in relations_to_create:
      existing, reversed = relation_to_create
      relations_dict[reversed] = self._reverse_relation(relations_dict[existing])

  def _reverse_relation(self, relation):
    if   relation == "child":
      return "parent"
    elif relation == "child_in_law":
      return "parent_in_law"
    elif relation == "parent":
      return "child"
    elif relation == "parent_in_law":
      return "child_in_law"
    elif relation == "sibling":
      return "sibling"
    elif relation == "sibling_in_law":
      return "sibling_in_law"
    elif relation == "grandparent":
      return "grandchild"
    elif relation == "grandchild":
      return "grandparent"
    elif relation == "nephew":
      return "uncle"
    elif relation == "uncle":
      return "nephew"
    elif relation == "so":
      return "so"
    elif relation == "son" or \
         relation == "daughter":
      return "parent"
    elif relation == "father" or \
         relation == "mother":
      return "child"
    elif relation == "grandson" or \
         relation == "granddaughter":
      return "grandparent"
    elif relation == "grandfather" or \
         relation == "grandmother":
      return "grandchild"
    elif relation == "uncle" or \
         relation == "aunt":
      return "nephew"
    elif relation == "son_in_law" or \
         relation == "daughter_in_law":
      return "parent_in_law"
    elif relation == "father_in_law" or \
         relation == "mother_in_law":
      return "child_in_law"
    elif relation == "nephew" or \
         relation == "niece":
      return "uncle"
    elif relation == "brother" or \
         relation == "sister":
      return "sibling"
    elif relation == "brother_in_law" or \
         relation == "sister_in_law":
      return "sibling_in_law"
    elif relation == "husband" or \
         relation == "wife":
      return "so"
    else:
      return relation

class RawCLUTRRValidationDataset(TorchDataset, ABC):
  def __init__(
    self
  ):
    super(TorchDataset, self).__init__()
    self.dataset_db = RawCLUTRRDatasetDatabase()
    self.dataset_db.initialize()

  @abstractmethod
  def __getitem__(self, idx):
    pass

  @abstractmethod
  def __len__(self):
    pass

class RawCLUTRRGenderNetValidationDataset(RawCLUTRRValidationDataset):
  def __init__(self):
    super().__init__()

  def __getitem__(self, idx):
    sentences, entity, gender = self.dataset_db.get_gender_net_sample(idx)
    sentences = term2list2(
      parse(
        next(sentences) + "."
      )
    )
    for i, sentence in enumerate(sentences):
      sentences[i] = Term("s", sentence.args[0], Term(sentence.args[1].functor.strip("\"")))
    sentences = list2term(sentences)

    return (sentences, entity), self._encode_gender_label(gender)

  def __len__(self):
    return self.dataset_db.get_length_gender_net()

  def _encode_gender_label(gender_label):
    return F.one_hot(torch.tensor((0 if gender_label == "male" else 1)), num_classes = 2).type(torch.FloatTensor)

class RawCLUTRRRelExtractValidationDataset(RawCLUTRRValidationDataset):
  def __init__(self):
    super().__init__()

  def __getitem__(self, idx):
    _, sentence, entity_1, entity_2, embedding_part_1, embedding_part_2, relation = self.dataset_db.get_rel_extract_sample(idx)

    return (sentence, entity_1, entity_2, bytes_to_tensor(embedding_part_1), bytes_to_tensor(embedding_part_2)), self._encode_relation_label(relation)

  def __len__(self):
    return self.dataset_db.get_length_rel_extract()

  def _encode_relation_label(relation_label):
    relation_number = (
      0 if relation_label == "child" else (
        1 if relation_label == "child_in_law" else (
          2 if relation_label == "parent" else (
            3 if relation_label == "parent_in_law" else (
              4 if relation_label == "sibling" else (
                5 if relation_label == "sibling_in_law" else (
                  6 if relation_label == "grandparent" else (
                    7 if relation_label == "grandchild" else (
                      8 if relation_label == "nephew" else (
                        9 if relation_label == "uncle" else 10
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
    return F.one_hot(torch.tensor(relation_number), num_classes = 11).type(torch.FloatTensor)

  def update_embedding(self, sentence, entity_1, entity_2, encoder):
    self.dataset_db.update_embedding_rel_extract(sentence, entity_1, entity_2, encoder)

  def update_embeddings(self, encoder):
    self.dataset_db.update_embeddings_rel_extract(encoder)
