from random import randint
from deepproblog.dataset import DataLoader
from deepproblog.examples.MNIST.data import MNISTOperator, MNIST_train, MNIST_test
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.optimizer import SGD
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
import torch
from deepproblog.dataset import NoiseMutatorDecorator, MutatingDataset
from deepproblog.query import Query
from problog.logic import Constant

SHUFFLE_SEED = 93891135229321951416666238953136246253198775800639367087882728959728265151654

def noise(_, query: Query):
    new_query = query.replace_output([Constant(randint(0, 18))])
    return new_query

def main(calibrate=False):
  dataset = MNISTOperator(
      dataset_name="train",
      function_name="addition_noisy",
      operator=sum,
      size=1,
	  seed=SHUFFLE_SEED
  )
  noisy_dataset = MutatingDataset(dataset, NoiseMutatorDecorator(0.2, noise))

  noisy_dataset_length = len(noisy_dataset)
  noisy_dataset_train = noisy_dataset.subset(round(0.8 * noisy_dataset_length))
  noisy_dataset_validation = noisy_dataset.subset(round(0.8 * noisy_dataset_length), noisy_dataset_length)
  queries = DataLoader(noisy_dataset_train, 2)
  validation_queries = DataLoader(noisy_dataset_validation, 2)

  network = MNIST_Net()
  networks_evolution_collectors = {}
  if calibrate == True:
    net = TemperatureScalingNetwork(network, "mnist_net", validation_queries, batching=True)
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector()
  else:
    net = Network(network, "mnist_net", batching=True)
  net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
  model = Model("models/noisy_addition.pl", [net])

  model.add_tensor_source("train", MNIST_train)
  model.add_tensor_source("test", MNIST_test)

  model.set_engine(ExactEngine(model))
  model.optimizer = SGD(model, 1e-3)

  train = train_model(model, queries, 1, networks_evolution_collectors, log_iter=100)

  test_set = MNISTOperator(
      dataset_name="test",
      function_name="addition_noisy",
      operator=sum,
      size=1,
  )
  noisy_test_set = MutatingDataset(test_set, NoiseMutatorDecorator(0.2, noise))

  return [train, get_confusion_matrix(model, noisy_test_set, verbose=1)]