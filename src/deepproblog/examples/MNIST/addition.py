import fire
from json import dumps

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.examples.MNIST.data import (
    MNIST_train,
    MNIST_test,
    addition,
    RawMNISTValidationDataset
)
from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.train import train_model
from deepproblog.utils import get_configuration, format_time_precise, config_to_string

import torch.utils.data

def main(
  i = 1,
  calibrate = False,
  calibrate_after_each_train_iteration = False,
  save_model_state = True,
  model_state_name = None,
  logging = False
):
  parameters = {
      "method": ["gm", "exact"],
      "N": [1, 2, 3],
      "pretrain": [0],
      "exploration": [False, True],
      "run": range(5),
  }
  configuration = get_configuration(parameters, i)
  torch.manual_seed(configuration["run"])
  name = "addition_" + config_to_string(configuration) + "_" + format_time_precise()
  print(name)

  train_set = addition(configuration["N"], "train")
  test_set = addition(configuration["N"], "test")
  batch_size = 2
  loader = DataLoader(train_set, batch_size, False)
  if calibrate == True:
    validation_loader_for_calibration = TorchDataLoader(RawMNISTValidationDataset(), batch_size)

  network = MNIST_Net()
  pretrain = configuration["pretrain"]
  if pretrain is not None and pretrain > 0:
      network.load_state_dict(
          torch.load("models/pretrained/all_{}.pth".format(configuration["pretrain"]))
      )
  networks_evolution_collectors = {}
  if calibrate == True:
      net = TemperatureScalingNetwork(network, "mnist_net", validation_loader_for_calibration, batching = True, calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
      networks_evolution_collectors["calibration_collector"] = NetworkECECollector()
  else:
      net = Network(network, "mnist_net", batching = True)
  net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

  model = Model("models/addition.pl", [net])
  if configuration["method"] == "exact":
      if configuration["exploration"] or configuration["N"] > 2:
          print("Not supported?")
          exit()
      model.set_engine(ExactEngine(model), cache = True)
  elif configuration["method"] == "gm":
      model.set_engine(
          ApproximateEngine(
              model, 1, geometric_mean, exploration = configuration["exploration"]
          )
      )
  model.add_tensor_source("train", MNIST_train)
  model.add_tensor_source("test", MNIST_test)

  train = train_model(model, loader, 1, networks_evolution_collectors, log_iter = 100, profile = 0)

  if logging == True:
    train.logger.comment(dumps(model.get_hyperparameters()))
    train.logger.comment(
      "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose = 0).accuracy())
    )
    train.logger.write_to_file("log/" + name)

  if calibrate == True:
    net.calibrate()

  if save_model_state:
    if model_state_name:
      model.save_state("snapshot/" + model_state_name + ".pth")
    else:
      model.save_state("snapshot/" + name + ".pth")

  return [train, get_confusion_matrix(model, test_set, verbose = 0)]

if __name__ == "__main__":
  fire.Fire(main)