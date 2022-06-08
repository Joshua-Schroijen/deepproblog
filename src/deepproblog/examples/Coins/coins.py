import fire
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Coins.data.dataset import train_dataset, test_dataset, RawCoinsNet1ValidationDataset, RawCoinsNet2ValidationDataset
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.train import train_model
from deepproblog.utils.standard_networks import smallnet
from deepproblog.utils.stop_condition import Threshold, StopOnPlateau

def split_train_set(train_set):
  train_set_length = len(train_set)
  rest_train_set = train_set.subset(round(0.8 * train_set_length))
  validation_set = train_set.subset(round(0.8 * train_set_length), train_set_length)
  return [rest_train_set, validation_set]

def main(
  calibrate = False,
  calibrate_after_each_train_iteration = False
):
  batch_size = 5
  if calibrate == True:
    rest_train_set, validation_set = split_train_set(train_dataset)
    train_loader = DataLoader(rest_train_set, batch_size)
    calibration_net1_valid_loader = TorchDataLoader(RawCoinsNet1ValidationDataset(validation_set), batch_size)
    calibration_net2_valid_loader = TorchDataLoader(RawCoinsNet2ValidationDataset(validation_set), batch_size)
  else:
    train_loader = DataLoader(train_dataset, batch_size)
  lr = 1e-4

  networks_evolution_collectors = {}
  coin_network1 = smallnet(num_classes = 2, pretrained = True)
  coin_network2 = smallnet(num_classes = 2, pretrained = True)
  if calibrate == True:
    coin_net1 = TemperatureScalingNetwork(coin_network1, "net1", calibration_net1_valid_loader, batching = True, calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
    coin_net2 = TemperatureScalingNetwork(coin_network2, "net2", calibration_net2_valid_loader, batching = True, calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector()
  else:
    coin_net1 = Network(coin_network1, "net1", batching = True)
    coin_net2 = Network(coin_network2, "net2", batching = True)
  coin_net1.optimizer = torch.optim.Adam(coin_network1.parameters(), lr = lr)
  coin_net2.optimizer = torch.optim.Adam(coin_network2.parameters(), lr = lr)

  model = Model("model.pl", [coin_net1, coin_net2])
  if calibrate == True:
    model.add_tensor_source("train", rest_train_set)
  else:
    model.add_tensor_source("train", train_dataset)
  model.add_tensor_source("test", test_dataset)
  model.set_engine(ExactEngine(model), cache = True)
  train_obj = train_model(
    model,
    train_loader,
    StopOnPlateau("Accuracy", warm_up = 10, patience = 10) | Threshold("Accuracy", 1.0, duration = 2),
    networks_evolution_collectors,
    log_iter = 100 // batch_size,
    test_iter = 100 // batch_size,
    test = lambda x: [("Accuracy", get_confusion_matrix(x, test_dataset).accuracy())],
    infoloss = 0.25
  )

  if calibrate:
    coin_net1.calibrate()
    coin_net2.calibrate()

  return [train_obj, get_confusion_matrix(model, test_dataset, verbose = 0)]

if __name__ == "__main__":
  fire.Fire(main)
