from pathlib import Path
import sqlite3
from torch.utils.data import Dataset as TorchDataset

from problog.logic import list2term, Term, Constant

class RawCLUTRRDatasetDatabase:
  def initialize(self):
    self.connection = sqlite3.connect(Path(__file__).parent / 'raw_CLUTRR_dataset.sqlite')
    self.cursor = self.connection.cursor()
    if not self._is_CLUTRR_samples_db_ready():
      self.cursor.execute("CREATE TABLE CLUTRR_rel_extract_raw_data ( sentence_id integer, entity integer )")
      self.cursor.execute("CREATE TABLE CLUTRR_gender_net_raw_data ( sentences text, swap boolean )")
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

  def _is_CLUTRR_samples_db_ready(self):
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'CLUTRR_rel_extract_raw_data';")
    CLUTRR_rel_extract_raw_data_table_exists = (self.cursor.fetchall() != [])
    self.cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table' AND tbl_name = 'CLUTRR_gender_net_raw_data';")
    CLUTRR_gender_net_raw_data_table_exists = (self.cursor.fetchall() != [])
    return (CLUTRR_rel_extract_raw_data_table_exists and \
            CLUTRR_gender_net_raw_data_table_exists)

  def _get_swap_label(self, X, Y):
    return X > Y

class RawCLUTRRGenderNetValidationDataset(TorchDataset):
    def __init__(
        self,
        CLUTRR_dataset,
    ):
        self.CLUTRR_dataset = CLUTRR_dataset

    def __getitem__(self, idx):
        #         0
        # >>> text
        # [s([1, 2],2 is a brother of 1), s([0, 1],1 is the father of 0)]
        # >>> ent
        # 2
        # >>> type(text)
        # <class 'problog.logic.Term'>
        # >>> type(ent)
        # <class 'problog.logic.Constant'>
        # >>>
        # deepproblog.utils.parse(text + ".")!!!!
        story = self.CLUTRR_dataset[idx]
        story_text = list2term(
          [Term(
            's',
            list2term(entities),
            Constant(" ".join(sentence))
          ) for entities, sentence in story.get_sentences()]
        )
        story_entity = Constant(story.get_entities())
        story_genders = story.get_genders()

        return (story_text, story_entity), story_genders
        # >>> self.genders
        # {0: 'male', 1: 'female', 2: 'male', 3: 'male', 4: 'female', 5: 'female', 6: 'male', 7: 'female', 8: 'female'}
        # >>> self.entities
        # {'louis': 0, 'ada': 1, 'ralph': 2, 'jeremy': 3, 'lakisha': 4, 'wanda': 5, 'jonathan': 6, 'darlene': 7, 'elisha': 8}
        # >>> type(self.genders)
        # <class 'dict'>
        # >>> type(self.entities)
       # <class 'dict'>
        # >>> self.text
        # [7, 'is', 2, "'", 's', 'daughter', '.', 2, 'is', 'a', 'brother', 'of', 1, '.', 2, 'is', 'the', 'father', 'of', 7, '.', 8, 'is', 7, "'", 's', 'sister', '.', 5, 'has', 'a', 'brother', 'named', 6, '.', 5, 'is', 'a', 'daughter', 'of', 4, '.', 4, 'is', 'a', 'sister', 'of', 3, '.', 1, 'is', 0, "'", 's', 'daughter', '.', 6, 'has', 'a', 'sister', 'named', 7, '.', 3, 'is', 'a', 'brother', 'of', 2, '.']
        # >>> type(self.text)
        # <class 'list'>
        # "7 is 2 ' s daughter . 2 is a brother of 1 . 2 is the father of 7 . 8 is 7 ' s sister . 5 has a brother named 6 . 5 is a daughter of 4 . 4 is a sister of 3 . 1 is 0 ' s daughter . 6 has a sister named 7 . 3 is a brother of 2 ."
        # >>> self.get_sentences()
        # [([2, 7], ['7', 'is', '2', "'", 's', 'daughter']), ([1, 2], ['2', 'is', 'a', 'brother', 'of', '1']), ([2, 7], ['2', 'is', 'the', 'father', 'of', '7']), ([8, 7], ['8', 'is', '7', "'", 's', 'sister']), ([5, 6], ['5', 'has', 'a', 'brother', 'named', '6']), ([4, 5], ['5', 'is', 'a', 'daughter', 'of', '4']), ([3, 4], ['4', 'is', 'a', 'sister', 'of', '3']), ([0, 1], ['1', 'is', '0', "'", 's', 'daughter']), ([6, 7], ['6', 'has', 'a', 'sister', 'named', '7']), ([2, 3], ['3', 'is', 'a', 'brother', 'of', '2'])]
        # >>> s =  self.get_sentences()
        # >>> s[0]
        # ([2, 7], ['7', 'is', '2', "'", 's', 'daughter'])
        # >>> s[1]
        # ([1, 2], ['2', 'is', 'a', 'brother', 'of', '1'])

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
