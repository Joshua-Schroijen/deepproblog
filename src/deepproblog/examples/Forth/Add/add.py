import fire
import random
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from problog.logic import Constant

from deepproblog.dataset import DataLoader, QueryDataset, NoiseMutatorDecorator, MutatingDatasetWithItems
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Forth import EncodeModule
from deepproblog.examples.Forth.Add.data.for_calibration import RawAddNeural1ValidationDataset, RawAddNeural2ValidationDataset, neural_dataloader_collate_fn
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
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
  test_queries = QueryDataset("data/train{}_test{}_test.txt".format(train, test))
  val = QueryDataset("data/train{}_test{}_dev.txt".format(train, test))
  if train_with_label_noise:
    label_noise = lambda _, q: q.replace_output([random.choice([Constant(0), Constant(1)]), random.choice([Constant(i) for i in range(0, 10)])])
    train_queries = MutatingDatasetWithItems(train_queries, NoiseMutatorDecorator(label_noise_probability, label_noise))
  # all_test = DataLoader(QueryDataset('data/tests.pl'), 16)
  raw_add_neural1_validation_dataset = RawAddNeural1ValidationDataset()
  raw_add_neural2_validation_dataset = RawAddNeural2ValidationDataset()

  net1 = EncodeModule(30, 50, 10, "tanh")
  net2 = EncodeModule(22, 10, 2, "tanh")
  networks_evolution_collectors = {}
  if calibrate == True:
    network1 = TemperatureScalingNetwork(net1, "neural1", TorchDataLoader(raw_add_neural1_validation_dataset, 50, collate_fn = neural_dataloader_collate_fn(10)), calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
    network2 = TemperatureScalingNetwork(net2, "neural2", TorchDataLoader(raw_add_neural2_validation_dataset, 50, collate_fn = neural_dataloader_collate_fn(2)), calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
    test_network1 = TemperatureScalingNetwork(net1, "neural1", TorchDataLoader(raw_add_neural1_validation_dataset, 50, collate_fn = neural_dataloader_collate_fn(10)), k = 1, calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
    test_network2 = TemperatureScalingNetwork(net2, "neural2", TorchDataLoader(raw_add_neural2_validation_dataset, 50, collate_fn = neural_dataloader_collate_fn(2)), k = 1, calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector()
  else:
    network1 = Network(net1, "neural1")
    network2 = Network(net2, "neural2")
    test_network1 = Network(net1, "neural1", k = 1)
    test_network2 = Network(net2, "neural2", k = 1)
  network1.optimizer = torch.optim.Adam(net1.parameters(), lr = 0.02)
  network2.optimizer = torch.optim.Adam(net2.parameters(), lr = 0.02)

  model = Model("choose.pl", [network1, network2])
  test_model = Model("choose.pl", [test_network1, test_network2])
  model.set_engine(ExactEngine(model), cache = True)
  test_model.set_engine(ExactEngine(test_model), cache = False)
  train_obj = train_model(
    model,
    DataLoader(train_queries, 50),
    40,
    networks_evolution_collectors,
    log_iter = 20,
    test = lambda x: [("Accuracy", get_confusion_matrix(test_model, val, verbose = 0).accuracy())],
    test_iter = 100,
  )

  if calibrate:
    network1.calibrate()
    network2.calibrate()
    test_network1.calibrate()
    test_network2.calibrate()

  if save_model_state:
    if model_state_name:
      model.save_state(f"snapshot/{model_state_name}.pth")
    else:
      model.save_state(f"snapshot/forth_add.pth")

  cm = get_confusion_matrix(test_model, test_queries, verbose = 0)

  return [train_obj, cm]

if __name__ == "__main__":
  fire.Fire(main)
