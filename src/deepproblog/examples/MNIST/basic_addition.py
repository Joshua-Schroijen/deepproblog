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

  batch_size = 2
  train_dataloader = DataLoader(train_dataset, batch_size, False)

  MNIST_net_pytorch = MNIST_Net()
  MNIST_net = Network(MNIST_net_pytorch, "mnist_net", batching = True)
  MNIST_net.optimizer = torch.optim.Adam(MNIST_net_pytorch.parameters(), lr = 1e-3)

  model = Model("models/addition.pl", [MNIST_net])
  model.set_engine(
    ApproximateEngine(
      model, 1, geometric_mean, exploration = configuration["exploration"]
    )
  )
  model.add_tensor_source("train", MNIST_train)
  model.add_tensor_source("test", MNIST_test)

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