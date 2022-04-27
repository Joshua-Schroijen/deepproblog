import fire
import torch

from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Forth import EncodeModule
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
  dev_queries = QueryDataset("data/train{}_test{}_dev.txt".format(train, test))
  test_queries = QueryDataset("data/train{}_test{}_test.txt".format(train, test))

  fc1 = EncodeModule(20, 20, 2)

  networks_evolution_collectors = {}
  if calibrate == True:
    fc1_network = TemperatureScalingNetwork(fc1, "swap_net", DataLoader(dev_queries, 16), optimizer = torch.optim.Adam(fc1.parameters(), 1.0))
    fc1_test_network = TemperatureScalingNetwork(fc1, "swap_net", DataLoader(dev_queries, 16), k = 1)
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
        ("Accuracy", get_confusion_matrix(test_model, dev_queries).accuracy())
    ],
  )
  return [train_obj, get_confusion_matrix(test_model, dev_queries, verbose = 0)]

if __name__ == "__main__":
  fire.Fire(main())