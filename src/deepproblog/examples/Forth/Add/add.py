import fire
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Forth import EncodeModule
from deepproblog.examples.Forth.Add.data.for_calibration import RawAddNeural1ValidationDataset, RawAddNeural2ValidationDataset
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.train import train_model

def main(
  calibrate = False,
  calibrate_after_each_train_iteration = False
):
  train = 2
  test = 8

  train_queries = QueryDataset("data/train{}_test{}_train.txt".format(train, test))
  test_queries = QueryDataset("data/train{}_test{}_test.txt".format(train, test))
  val = QueryDataset("data/train{}_test{}_dev.txt".format(train, test))
  # all_test = DataLoader(QueryDataset('data/tests.pl'), 16)

  net1 = EncodeModule(30, 50, 10, "tanh")
  net2 = EncodeModule(22, 10, 2, "tanh")

  networks_evolution_collectors = {}
  if calibrate == True:
    network1 = TemperatureScalingNetwork(net1, "neural1", TorchDataLoader(RawAddNeural1ValidationDataset(), 50))
    network2 = TemperatureScalingNetwork(net2, "neural2", TorchDataLoader(RawAddNeural2ValidationDataset(), 50))
    test_network1 = TemperatureScalingNetwork(net1, "neural1", TorchDataLoader(RawAddNeural1ValidationDataset(), 50), k = 1)
    test_network2 = TemperatureScalingNetwork(net2, "neural2", TorchDataLoader(RawAddNeural2ValidationDataset(), 50), k = 1)
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
  cm = get_confusion_matrix(test_model, val, verbose = 0)
  return [train_obj, cm]

if __name__ == "__main__":
  fire.Fire(main)
