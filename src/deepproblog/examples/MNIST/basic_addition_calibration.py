import torch
from torch.utils.data import DataLoader as TorchDataLoader
from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.model import Model
from deepproblog.train import train_model
from deepproblog.examples.MNIST.data import MNISTOperator, MNIST_train, MNIST_test, RawMNISTValidationDataset
from deepproblog.examples.MNIST.network import MNIST_Net

if __name__ == "__main__":
  # General DeepProbLog flow - step 1
  # Load and create dataset objects (of (sub)type deepproblog.dataset.Dataset)
  train_dataset = MNISTOperator(
    dataset_name = "train",
    function_name = "addition",
    operator = sum,
    size = 1,
    arity = 2
  )
  test_dataset = MNISTOperator(
    dataset_name = "test",
    function_name = "addition",
    operator = sum,
    size = 1,
    arity = 2
  )

  # General DeepProbLog flow - step 2
  # Create data loader objects based on the dataset objects (of (sub)type deepproblog.dataset.DataLoader)
  batch_size = 2
  train_dataloader = DataLoader(train_dataset, batch_size, False)

  # For calibrating the model, create a regular PyTorch data loader
  # based on a raw NN dataset. Note that we use data that is no longer
  # used in training or test queries, because this is a requirement of
  # temperature scaling.
  validation_loader_for_calibration = TorchDataLoader(RawMNISTValidationDataset(), batch_size)
  validation_loader_for_calibration = TorchDataLoader(RawMNISTValidationDataset(), batch_size)

  # General DeepProbLog flow - step 3
  # Create PyTorch NN objects
  MNIST_net_pytorch = MNIST_Net()

  # General DeepProbLog flow - step 4
  # Create DeepProbLog network objects (type deepproblog.network.Network) based on the PyTorch NNs
  # For calibrating the model, use subclass of subclass
  # deepproblog.calibrated_network.CalibratedNetwork, such as
  # deepproblog.calibrated_network.TemperatureScalingNetwork to apply temperature scaling,
  # and define a deepproblog.calibrated_network.NetworkECECollector to monitor model NN ECE evolution
  # as a deepproblog.networks_evolution_collector.NetworksEvolutionCollector
  # during training.
  networks_evolution_collectors = {}
  MNIST_net = TemperatureScalingNetwork(MNIST_net_pytorch, "mnist_net", validation_loader_for_calibration, batching = True, calibrate_after_each_train_iteration = False)
  MNIST_net.optimizer = torch.optim.Adam(MNIST_net_pytorch.parameters(), lr = 1e-3)
  networks_evolution_collectors["calibration_collector"] = NetworkECECollector(
    {"mnist_net": validation_loader_for_calibration},
    iteration_collect_iter = 100
  )

  # General DeepProbLog flow - step 5
  # Construct a DeepProbLog model object (type deepproblog.model.Model) based on the KB file and the DeepProbLog Network objects
  model = Model("models/addition.pl", [MNIST_net])

  # General DeepProbLog flow - step 6
  # Create an engine object and add it to the model. An approximate (class deepproblog.engines.ApproximateEngine) and exact (deepproblog.engines.ExactEngine) inference engine are provided in the standard DeepProbLog distribution. Both engines have cases in which they are or are not appropriate.
  model.set_engine(ExactEngine(model), cache = True)

  # General DeepProbLog flow - step 7
  # Couple tensor source objects to the model using its add_tensor_source method
  model.add_tensor_source("train", MNIST_train)
  model.add_tensor_source("test", MNIST_test)

  # General DeepProbLog flow - step 8
  # Use the deepproblog.train.train_model function to train the DeepProbLog model (which means optimizing the unknown model probabilities/parameters and the model's NNs' weights for model accuracy on a test set of queries)
  # For calibrating the model, we pass along the networks evolution collectors
  # to monitor model NN ECE evolution during training
  train = train_model(
    model,
    train_dataloader,
    1,
    networks_evolution_collectors,
    log_iter = 100,
    profile = 0
  )

  model.save_state(f"snapshot/basic_MNIST_model.pth")

  accuracy = get_confusion_matrix(model, test_dataset, verbose = 0).accuracy()
  print(f"Done.\nThe model acccuracy was {accuracy}.")

  for networks_evolution_collector in train.networks_evolution_collectors.values():
    ece_histories = networks_evolution_collector.collection_as_dict()["ece_history"]
    for network_name in ece_histories:
      initial_ECE = ece_histories[network_name][0]
      final_ECE = ece_histories[network_name][-1]
      print(f"Model NN {network_name} initial ECE was {initial_ECE}")
      print(f"Model NN {network_name} final ECE was {final_ECE}")
