import fire
import random
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from problog.logic import list2term, term2list

from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.dataset import DataLoader, QueryDataset, NoiseMutatorDecorator, MutatingDatasetWithItems
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Forth.Sort.data.for_calibration import RawSortValidationDataset, sort_dataloader_collate_fn
from deepproblog.examples.Forth import EncodeModule
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.query import Query
from deepproblog.train import train_model

def main(
  calibrate = False,
  calibrate_after_each_train_iteration = False,
  save_model_state = True,
  model_state_name = None,
  train_with_label_noise = False,
  label_noise_probability = 0.2,
):
  train = 2
  test = 8
  train_queries = QueryDataset("data/train{}_test{}_train.txt".format(train, test))
  dev_queries = QueryDataset("data/train{}_test{}_dev.txt".format(train, test))
  test_queries = QueryDataset("data/train{}_test{}_test.txt".format(train, test))
  if train_with_label_noise:
    label_noise = lambda _, q: q.replace_output([list2term(sorted(term2list(q.query.args[0]), reverse = True))])
    train_queries = MutatingDatasetWithItems(train_queries, NoiseMutatorDecorator(label_noise_probability, label_noise))
  raw_validation_dataset = RawSortValidationDataset()
  
  fc1 = EncodeModule(20, 20, 2)
  networks_evolution_collectors = {}
  if calibrate == True:
    fc1_network = TemperatureScalingNetwork(fc1, "swap_net", TorchDataLoader(raw_validation_dataset, 16, collate_fn = sort_dataloader_collate_fn), optimizer = torch.optim.Adam(fc1.parameters(), 1.0), calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
    fc1_test_network = TemperatureScalingNetwork(fc1, "swap_net", TorchDataLoader(raw_validation_dataset, 16, collate_fn = sort_dataloader_collate_fn), k = 1, calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector()
  else:
    fc1_network = Network(fc1, "swap_net", optimizer = torch.optim.Adam(fc1.parameters(), 1.0))
    fc1_test_network = Network(fc1, "swap_net", k = 1)

  model = Model("compare.pl", [fc1_network])
  model.set_engine(ExactEngine(model), cache = True)
  test_model = Model("compare.pl", [fc1_test_network])
  test_model.set_engine(ExactEngine(test_model), cache = False)

  train_obj = train_model(
    model,
    DataLoader(train_queries, 16),
    40,
    networks_evolution_collectors,
    log_iter = 50,
    test_iter = len(train_queries),
    test = lambda x: [
      ("Accuracy", get_confusion_matrix(test_model, dev_queries, verbose = 0).accuracy())
    ],
  )

  if calibrate:
    fc1_network.calibrate()
    fc1_test_network.calibrate()

  if save_model_state:
    if model_state_name:
      model.save_state(f"snapshot/{model_state_name}.pth")
    else:
      model.save_state(f"snapshot/forth_sort.pth")

  return [train_obj, get_confusion_matrix(model, test_queries, verbose = 0)]

if __name__ == "__main__":
  fire.Fire(main)