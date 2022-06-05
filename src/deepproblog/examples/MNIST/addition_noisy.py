from random import randint
from typing import Tuple

from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.dataset import DataLoader
from deepproblog.examples.MNIST.data import MNISTOperator, MNIST_train, MNIST_test
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.optimizer import SGD
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.dataset import NoiseMutatorDecorator, MutatingDataset
from deepproblog.query import Query
from deepproblog.utils import MutatingRawDataset

from problog.logic import Constant

import torch
from torch.utils.data import DataLoader as TorchDataLoader

SHUFFLE_SEED = 93891135229321951416666238953136246253198775800639367087882728959728265151654

def noise(_, query: Query):
  new_query = query.replace_output([Constant(randint(0, 18))])
  return new_query

def noise_raw(_, item: Tuple[torch.Tensor, torch.Tensor, int]):
  return (item[0], item[1], randint(0, 18))

def main(
  calibrate = False
):
  dataset = MNISTOperator(
    dataset_name = "train",
    function_name = "addition_noisy",
    operator = sum,
    size = 1,
	  seed = SHUFFLE_SEED
  )
  noisy_dataset = MutatingDataset(dataset, NoiseMutatorDecorator(0.2, noise))

  noisy_dataset_length = len(noisy_dataset)
  noisy_dataset_train = noisy_dataset.subset(round(0.8 * noisy_dataset_length))
  noisy_dataset_validation = noisy_dataset.subset(round(0.8 * noisy_dataset_length), noisy_dataset_length)
  queries = DataLoader(noisy_dataset_train, 2)

  network = MNIST_Net()
  networks_evolution_collectors = {}
  if calibrate == True:
    validation_loader = TorchDataLoader(MutatingRawDataset(noisy_dataset_validation, noise_raw), 2)
    net = TemperatureScalingNetwork(network, "mnist_net", validation_loader, batching = True)
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector()
  else:
    net = Network(network, "mnist_net", batching = True)
  net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
  model = Model("models/noisy_addition.pl", [net])

  model.add_tensor_source("train", MNIST_train)
  model.add_tensor_source("test", MNIST_test)

  model.set_engine(ExactEngine(model))
  model.optimizer = SGD(model, 1e-3)

  train = train_model(model, queries, 1, networks_evolution_collectors, log_iter = 100)

  test_set = MNISTOperator(
      dataset_name = "test",
      function_name = "addition_noisy",
      operator = sum,
      size = 1,
  )
  noisy_test_set = MutatingDataset(test_set, NoiseMutatorDecorator(0.2, noise))

  if calibrate:
    net.calibrate()

  return [train, get_confusion_matrix(model, noisy_test_set, verbose=1)]

if __name__ == "__main__":
  main()