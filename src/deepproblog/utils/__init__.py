from datetime import datetime
import io
import os
import random
import sys
from configparser import ConfigParser
from datetime import datetime
from itertools import islice
from pathlib import Path
from statistics import mean, stdev
from time import strftime
from typing import Union, Any, Dict, List

import torch
from torch.utils.data import Dataset as TorchDataset, Subset as TorchSubset

import problog
from problog.logic import Term

from ..dataset import Dataset

parser = problog.parser.PrologParser(problog.program.ExtendedPrologFactory())

cred = "\033[91m"
cend = "\033[0m"
cgreen = "\033[92m"

def print_with_timestamp(to_print: str):
    now = datetime.now()

    current_time = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"{current_time}: {to_print}")

def log_exists(location: Union[str, os.PathLike], name: str):
    return Path(location).glob(name + "*")

def check_path(path: Union[str, os.PathLike]):
    path_dir = os.path.dirname(str(path))
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

def get_top_path(pattern: str, reverse=True):
    paths = sorted(Path(".").glob(pattern), reverse=reverse)
    if len(paths) > 0:
        return paths[0]
    return None

def split(text: str, splitchar: str, lb="(", rb=")"):
    depth = 0
    splits = [""]
    for c in text:
        if c in splitchar and depth == 0:
            splits.append("")
            continue
        if c in lb:
            depth += 1
        if c in rb:
            depth -= 1
        splits[-1] += c
    return splits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def nth(iterable, n, default=None):
    """Returns the nth item or a default value"""
    return next(islice(iterable, n, None), default)

def get_configuration(parameters: dict, i: int) -> dict:
    config = dict()
    for k in parameters:
        l = len(parameters[k])
        j, i = i % l, i // l
        config[k] = parameters[k][j]
    return config

def parse(string: str) -> Term:
    parsed = parser.parseString(string)
    for term in parsed:
        return term

class Table(object):
    class Dimension(object):
        def __init__(self, name):
            self.name = name
            self.type = "categorical"
            self.categories = set()

        def add_value(self, val):
            self.categories.add(val)

        def __repr__(self):
            return self.name

    def __init__(self, *dimensions):
        self.data = []
        self.dimensions = [Table.Dimension(d) for d in dimensions]

    def store(self, *args, **kwargs):
        if len(args) == 0:
            data = tuple(kwargs[str(d)] for d in self.dimensions)
        else:
            if len(args) != len(self.dimensions):
                raise ValueError(
                    "Data dimensions {} not equal to table dimensions {}".format(
                        len(args), len(self.dimensions)
                    )
                )
            data = args
        self.data.append(data)
        for i, d in enumerate(self.dimensions):
            d.add_value(data[i])

    def get_dimension_index(self, dim):
        for i, d in enumerate(self.dimensions):
            if hash(dim) == hash(d):
                return i
        raise ValueError("{} not in dimensions".format(dim))

    def aggregate(self, l):
        if l is None or len(l) == 0:
            return ""
        mu, sig = mean(l), stdev(l)
        return "{:.5f} ± {:.5f}".format(mu, sig)

    def format(self, x, y, val, **kwargs):
        categories = tuple(kwargs.items())
        categories = tuple((self.get_dimension_index(c[0]), c[1]) for c in categories)
        x = self.get_dimension_index(x)
        y = self.get_dimension_index(y)
        val = self.get_dimension_index(val)
        x_cats = list(self.dimensions[x].categories)
        y_cats = list(self.dimensions[y].categories)
        data = [[None] * len(x_cats) for _ in y_cats]
        for d in self.data:
            j = x_cats.index(d[x])
            i = y_cats.index(d[y])
            correct_categories = True
            for k, v in categories:
                if d[k] != v:
                    correct_categories = False
                    break
            if correct_categories:
                if data[i][j] is None:
                    data[i][j] = []
                data[i][j].append(d[val])
        data = [[self.aggregate(d) for d in row] for row in data]
        return TabularFormatter.format(data, x_cats, y_cats)

class TabularFormatter(object):
    @staticmethod
    def format(data, x=None, y=None):
        if y is not None:
            data = [[y[i]] + data[i] for i in range(len(data))]
            if x is not None:
                data = [[""] + x] + data
        else:
            if x is not None:
                data = x + data

        nr_columns = len(data[0])
        column_widths = [0] * nr_columns
        for row in data:
            for i, value in enumerate(row):
                column_widths[i] = max(column_widths[i], len(str(value)))

        padded_rows = [
            "\t".join(
                [
                    " " * (column_widths[i] - len(str(v))) + str(v)
                    for i, v in enumerate(row)
                ]
            )
            for row in data
        ]
        return "\n".join(padded_rows)

def format_time():
    return strftime("_%y%m%d_%H%M")

def format_time_precise():
    return datetime.utcnow().strftime("%y%m%d_%H%M%S%f")

class NoConfigException(Exception):
    def __str__(self):
        return "No config file specified as an argument."

def load_config(filename: str = None):
    """
    Loads a config file.
    :param filename: Filename of configuration file to load. If None, it uses the first commandline argument as filename.
    :return: None
    """
    try:
        if filename is None:
            filename = sys.argv[1]
        config = ConfigParser()
        config.read(filename)
        return config["Default"]
    except IndexError:
        raise NoConfigException()

def term2list2(term: Term):
    result = []
    while (
        not problog.logic.is_variable(term) and term.functor == "." and term.arity == 2
    ):
        result.append(term.args[0])
        term = term.args[1]
    if not term == problog.logic.Term("[]"):
        raise ValueError("Expected fixed list.")
    return result

def config_to_string(configuration: Dict[str, Any]) -> str:
    return "_".join(
        "{}_{}".format(parameter, configuration[parameter])
        for parameter in configuration
    )

def tensor_to_bytes(tensor):
  buffer = io.BytesIO()
  torch.save(tensor, buffer)
  buffer.seek(0)
  return buffer.read()

def bytes_to_tensor(blob):
  buffer = io.BytesIO()
  buffer.write(blob)
  buffer.seek(0)
  return torch.load(buffer)

def random_one_hot_tensor(numel):
  one_hot = torch.zeros(numel, dtype = torch.float32)
  hot_index = random.randint(0, numel - 1)
  one_hot[hot_index] = 1
  return one_hot

class MutatingRawDataset(TorchDataset):
  def __init__(self, inner_raw_dataset, mutator, p, seed = None):
    super(TorchDataset, self).__init__()
    self.inner_raw_dataset = inner_raw_dataset
    self.mutator = mutator
    self.p = p
    if seed is None:
        seed = random.SystemRandom().randint(0, 2 ** 64)
    self.seed = seed

  def __len__(self):
    return len(self.inner_raw_dataset)

  def __getitem__(self, idx):
    rng = random.Random(self.seed ^ idx)
    if rng.random() < self.p:
      return self.mutator(idx, self.inner_raw_dataset[idx])
    else:
      return self.inner_raw_dataset[idx]

def split_dataset(dataset: Dataset, split_ratio: float = 0.8):
  dataset_length = len(dataset)
  dataset_part_1 = dataset.subset(round(split_ratio * dataset_length))
  dataset_part_2 = dataset.subset(round(split_ratio * dataset_length), dataset_length)
  return [dataset_part_1, dataset_part_2]

def split_torch_dataset(dataset: TorchDataset, split_ratio: float = 0.8):
  dataset_length = len(dataset)
  part_1_indices = range(round(split_ratio * dataset_length))
  part_2_indices = range(round(split_ratio * dataset_length), dataset_length)
  dataset_part_1 = TorchSubset(dataset, part_1_indices)
  dataset_part_2 = TorchSubset(dataset, part_2_indices)
  return [dataset_part_1, dataset_part_2]

def load_list(the_list: List[Any], batch_size: int):
    if batch_size < 1:
        return []
    start = 0
    end = batch_size
    while start < len(the_list):
        yield the_list[start:end]
        start += batch_size
        end += batch_size