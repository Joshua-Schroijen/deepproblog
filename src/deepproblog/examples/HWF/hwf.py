import fire
from json import dumps
import random
from sys import argv

from torch.optim import Adam
from torch.utils.data import DataLoader as TorchDataLoader

from problog.logic import Constant

from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.dataset import DataLoader, NoiseMutatorDecorator, MutatingDatasetWithItems 
from deepproblog.train import train_model
from deepproblog.examples.HWF.data import HWFDataset, hwf_images
from deepproblog.examples.HWF.data.for_calibration import RawHWFDatasetDatabase, RawHWFNumbersValidationDataset, RawHWFOperatorsValidationDataset
from deepproblog.examples.HWF.network import SymbolEncoder, SymbolClassifier
from deepproblog.heuristics import *
from deepproblog.utils import format_time_precise, get_configuration, config_to_string

def main(
  i = 0,
  calibrate = False,
  calibrate_after_each_train_iteration = False,
  save_model_state = True,
  model_state_name = None,
  train_with_label_noise = False,
  label_noise_probability = 0.2,
  logging = False
):
  configurations = {
    "method": ["exact"],
    "curriculum": [False],
    "N": [1, 3],
    "run": range(5),
  }
  configuration = get_configuration(configurations, i)
  name = "hwf_" + config_to_string(configuration) + "_" + format_time_precise()
  torch.manual_seed(configuration["run"])
  N = configuration["N"]
  if configuration["method"] == "exact":
    if N > 3:
      exit()

  curriculum = configuration["curriculum"]
  print("Training HWF with N={} and curriculum={}".format(N, curriculum))

  if curriculum:
    dataset_filter = lambda x: x <= N
    calibration_validation_dataset_filter = lambda x: x <= max(N, 3)
    dataset = HWFDataset("train2", dataset_filter)
    val_dataset = HWFDataset("val", calibration_validation_dataset_filter)
    test_dataset = HWFDataset("test", dataset_filter)
  else:
    dataset_filter = lambda x: x == N
    calibration_validation_dataset_filter = lambda x: x == max(N, 3)
    dataset = HWFDataset("train2", dataset_filter)
    val_dataset = HWFDataset("val", calibration_validation_dataset_filter)
    test_dataset = HWFDataset("test", dataset_filter)
  if train_with_label_noise:
    label_noise = lambda _, q: q.replace_output([Constant(random.uniform(-450, 7000))])
    dataset = MutatingDatasetWithItems(dataset, NoiseMutatorDecorator(label_noise_probability, label_noise))
  loader = DataLoader(dataset, 32, shuffle = True)

  encoder = SymbolEncoder()
  network1 = SymbolClassifier(encoder, 10)
  network2 = SymbolClassifier(encoder, 4)
  networks_evolution_collectors = {}
  if calibrate == True:
    raw_hwf_dataset_database = RawHWFDatasetDatabase()
    raw_hwf_dataset_database.initialize(calibration_validation_dataset_filter)
    raw_hwf_numbers_validation_dataset = RawHWFNumbersValidationDataset(raw_hwf_dataset_database)
    raw_hwf_operators_validation_dataset = RawHWFOperatorsValidationDataset(raw_hwf_dataset_database)
    net1_valid_loader = TorchDataLoader(raw_hwf_numbers_validation_dataset, 32, shuffle = True)
    net2_valid_loader = TorchDataLoader(raw_hwf_operators_validation_dataset, 32, shuffle = True)
    net1 = TemperatureScalingNetwork(network1, "net1", net1_valid_loader, Adam(network1.parameters(), lr = 3e-3), batching = True, calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
    net2 = TemperatureScalingNetwork(network2, "net2", net2_valid_loader, Adam(network2.parameters(), lr = 3e-3), batching = True, calibrate_after_each_train_iteration = calibrate_after_each_train_iteration)
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector(
      {"net1": net1_valid_loader, "net2": net2_valid_loader},
      iteration_collect_iter = 25
    )
  else:
    net1 = Network(network1, "net1", Adam(network1.parameters(), lr = 3e-3), batching = True)
    net2 = Network(network2, "net2", Adam(network2.parameters(), lr = 3e-3), batching = True)

  model = Model("model.pl", [net1, net2])
  model.add_tensor_source("hwf", hwf_images)
  heuristic = GeometricMean()
  if configuration["method"] == "exact":
    model.set_engine(ExactEngine(model), cache = True)
  elif configuration["method"] == "approximate":
    model.set_engine(
      ApproximateEngine(
        model, 1, heuristic, timeout = 30, ignore_timeout = True, exploration = True
      )
    )

  print("Training on size {}".format(N))
  train_log = train_model(
    model,
    loader,
    50,
    networks_evolution_collectors,
    log_iter = 50,
    inital_test = False,
    test_iter = 100,
    test = lambda x: [
      ("Val_accuracy", get_confusion_matrix(x, val_dataset, eps = 1e-6).accuracy()),
      ("Test_accuracy", get_confusion_matrix(x, test_dataset, eps = 1e-6).accuracy()),
    ],
  )

  ECEs_final_calibration = {
    "net1": {},
    "net2": {}
  }
  if calibrate == True:
    ECEs_final_calibration["net1"]["before"] = net1.get_expected_calibration_error(net1_valid_loader) 
    ECEs_final_calibration["net2"]["before"] = net2.get_expected_calibration_error(net2_valid_loader) 
    net1.calibrate()
    net2.calibrate()
    ECEs_final_calibration["net1"]["after"] = net1.get_expected_calibration_error(net1_valid_loader) 
    ECEs_final_calibration["net2"]["after"] = net2.get_expected_calibration_error(net2_valid_loader) 

  cm = get_confusion_matrix(model, test_dataset, eps = 1e-6, verbose = 0)
  final_acc = cm.accuracy()
  if logging == True:
    train_log.logger.comment("Accuracy {}".format(final_acc))
    train_log.logger.comment(dumps(model.get_hyperparameters()))
    train_log.write_to_file("log/" + name)

  if save_model_state:
    if model_state_name:
      model.save_state("snapshot/" + model_state_name + ".pth")
    else:
      model.save_state("snapshot/" + name + ".pth")

  return [train_log, cm, ECEs_final_calibration]

if __name__ == "__main__":
  fire.Fire(main)