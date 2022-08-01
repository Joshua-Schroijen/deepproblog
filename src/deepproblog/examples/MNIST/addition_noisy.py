import fire
from random import randint

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from problog.logic import Constant

from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.dataset import DataLoader
from deepproblog.examples.MNIST.data import MNISTOperator, RawMNISTValidationDataset, MNIST_train, MNIST_test, MNIST_raw_noise
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

SHUFFLE_SEED = 93891135229321951416666238953136246253198775800639367087882728959728265151654

def noise(_, query: Query):
  new_query = query.replace_output([Constant(randint(0, 18))])
  return new_query

def main(
  calibrate = False,
  calibrate_after_each_train_iteration = False,
  save_model_state = True,
  model_state_name = None,
):
  dataset = MNISTOperator(
    dataset_name = "train",
    function_name = "addition_noisy",
    operator = sum,
    size = 1,
	  seed = SHUFFLE_SEED
  )
  noisy_dataset_train = MutatingDataset(dataset, NoiseMutatorDecorator(0.2, noise))
  queries = DataLoader(noisy_dataset_train, 2)
  noisy_dataset_validation = MutatingRawDataset(RawMNISTValidationDataset(), MNIST_raw_noise, 0.2)
  test_set = MNISTOperator(
      dataset_name = "test",
      function_name = "addition_noisy",
      operator = sum,
      size = 1,
  )
  noisy_test_set = MutatingDataset(test_set, NoiseMutatorDecorator(0.2, noise))

  network = MNIST_Net()
  networks_evolution_collectors = {}
  if calibrate == True:
    validation_loader = TorchDataLoader(noisy_dataset_validation, 2)
    net = TemperatureScalingNetwork(network, "mnist_net", validation_loader, batching = True, calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector(
      {"mnist_net": validation_loader},
      iteration_collect_iter = 100
    )
  else:
    net = Network(network, "mnist_net", batching = True)
  net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
  model = Model("models/noisy_addition.pl", [net])

  model.add_tensor_source("train", MNIST_train)
  model.add_tensor_source("test", MNIST_test)

  model.set_engine(ExactEngine(model))
  model.optimizer = SGD(model, 1e-3)

  train = train_model(model, queries, 1, networks_evolution_collectors, log_iter = 100)

  ECEs_final_calibration = {
    "mnist_net": {}
  }
  if calibrate:
    ECEs_final_calibration["mnist_net"]["before"] = net.get_expected_calibration_error(validation_loader)    
    net.calibrate()
    ECEs_final_calibration["mnist_net"]["after"] = net.get_expected_calibration_error(validation_loader)

  if save_model_state:
    if model_state_name:
      model.save_state("snapshot/" + model_state_name + ".pth")
    else:
      model.save_state("snapshot/addition_noisy.pth")

  return [train, get_confusion_matrix(model, noisy_test_set, verbose = 0), ECEs_final_calibration]

if __name__ == "__main__":
  fire.Fire(main)