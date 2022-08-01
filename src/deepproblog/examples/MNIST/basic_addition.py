import torch
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.examples.MNIST.data import MNISTOperator, MNIST_train, MNIST_test
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

  # General DeepProbLog flow - step 3
  # Create PyTorch NN objects
  MNIST_net_pytorch = MNIST_Net()

  # General DeepProbLog flow - step 4
  # Create DeepProbLog network objects (type deepproblog.network.Network) based on the PyTorch NNs
  MNIST_net = Network(MNIST_net_pytorch, "mnist_net", batching = True)
  MNIST_net.optimizer = torch.optim.Adam(MNIST_net_pytorch.parameters(), lr = 1e-3)

  # General DeepProbLog flow - step 5
  # Construct a DeepProbLog model object (type deepproblog.model.Model) based on the KB file and the DeepProbLog Network objects
  model = Model("models/addition.pl", [MNIST_net])

  # General DeepProbLog flow - step 6
  # Create an engine object and add it to the model. An approximate (class deepproblog.engines.ApproximateEngine) and exact (deepproblog.engines.ExactEngine) inference engine are provided in the standard DeepProbLog distribution. Both engines have cases in which they are or are not appropriate.
  model.set_engine(
    ApproximateEngine(
      model, 1, geometric_mean, exploration = False
    )
  )

  # General DeepProbLog flow - step 7
  # Couple tensor source objects to the model using its add_tensor_source method
  model.add_tensor_source("train", MNIST_train)
  model.add_tensor_source("test", MNIST_test)

  # General DeepProbLog flow - step 8
  # Use the deepproblog.train.train_model function to train the DeepProbLog model (which means optimizing the unknown model probabilities/parameters and the model's NNs' weights for model accuracy on a test set of queries)
  train = train_model(
    model,
    train_dataloader,
    1,
    log_iter = 100,
    profile = 0
  )

  model.save_state(f"snapshot/basic_MNIST_model.pth")

  accuracy = get_confusion_matrix(model, test_dataset, verbose = 0).accuracy()
  print(f"Done. The model acccuracy was {accuracy}.")