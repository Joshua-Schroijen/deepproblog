import fire
import random
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from problog.logic import Constant
from deepproblog.dataset import DataLoader, NoiseMutatorDecorator, MutatingDatasetWithItems
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Poker import PokerSeparate, RawPokerNet1ValidationDataset
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.optimizer import SGD
from deepproblog.train import train_model
from deepproblog.utils import split_dataset
from deepproblog.utils.standard_networks import smallnet

def main(
  calibrate = False,
  calibrate_after_each_train_iteration = False,
  save_model_state = True,
  model_state_name = None,
  train_with_label_noise = False,
  label_noise_probability = 0.2,
):
  datasets = {
    "unfair": PokerSeparate(
      "unfair", probs = [0.2, 0.4, 0.15, 0.25], extra_supervision = True
    ),
    "fair_test": PokerSeparate("fair_test"),
  }
  dataset = "unfair"

  if train_with_label_noise:
    label_noise = lambda _, q: q.replace_output([random.choice([Constant("win"), Constant("loss"), Constant("draw")])])
    datasets["unfair"] = MutatingDatasetWithItems(datasets["unfair"], NoiseMutatorDecorator(label_noise_probability, label_noise))

  batch_size = 50
  if calibrate == True:
    rest_train_set, validation_set = split_dataset(datasets[dataset])
    train_loader = DataLoader(rest_train_set, batch_size)
    calibration_valid_loader = TorchDataLoader(RawPokerNet1ValidationDataset(validation_set), batch_size)
  else:
    train_loader = DataLoader(datasets[dataset], batch_size)

  networks_evolution_collectors = {}
  if calibrate == True:
    net = TemperatureScalingNetwork(
      smallnet(pretrained = True, num_classes = 4, size = (100, 150)),
      "net1",
      calibration_valid_loader,
      batching = True,
      calibrate_after_each_train_iteration = calibrate_after_each_train_iteration
    )
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector({"net1": calibration_valid_loader})
  else:
    net = Network(
      smallnet(pretrained = True, num_classes = 4, size = (100, 150)),
      "net1",
      batching = True
    )
  net.optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)

  model = Model("model.pl", [net])
  model.set_engine(ExactEngine(model), cache = True)
  model.optimizer = SGD(model, 5e-2)
  model.add_tensor_source(dataset, datasets[dataset])
  model.add_tensor_source("fair_test", datasets["fair_test"])

  train_obj = train_model(
    model,
    train_loader,
    10,
    networks_evolution_collectors,
    loss_function_name = "mse",
    log_iter = len(datasets["unfair"]) // batch_size,
    test_iter = 5 * len(datasets["unfair"]) // batch_size,
    test = lambda x: [
      ("Accuracy", get_confusion_matrix(model, datasets["fair_test"]).accuracy())
    ],
    infoloss = 0.5,
  )

  ECEs_final_calibration = {
    "net1": {}
  }
  if calibrate == True:
    ECEs_final_calibration["net1"]["before"] = net.get_expected_calibration_error(calibration_valid_loader)
    net.calibrate()
    ECEs_final_calibration["net1"]["after"] = net.get_expected_calibration_error(calibration_valid_loader)

  if save_model_state:
    if model_state_name:
      model.save_state("snapshot/" + model_state_name + ".pth")
    else:
      model.save_state("snapshot/poker.pth")

  cm = get_confusion_matrix(model, datasets["fair_test"], verbose = 0)
  return [train_obj, cm, ECEs_final_calibration]

if __name__ == "__main__":
  fire.Fire(main)