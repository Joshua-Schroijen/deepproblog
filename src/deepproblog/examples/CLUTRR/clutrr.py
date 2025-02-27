import fire
from json import dumps
import random
from torch.utils.data import DataLoader as TorchDataLoader

from problog.logic import Constant

from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.dataset import DataLoader, NoiseMutatorDecorator, MutatingDatasetWithItems
from deepproblog.engines import ApproximateEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.CLUTRR.architecture import Encoder, RelNet, GenderNet
from deepproblog.examples.CLUTRR.data import CLUTRR, dataset_names
from deepproblog.examples.CLUTRR.data.for_calibration import RawCLUTRRRelExtractValidationDataset, RawCLUTRRGenderNetValidationDataset, gender_net_dataloader_collate_fn, rel_extract_dataloader_collate_fn
from deepproblog.network import Network
from deepproblog.model import Model
from deepproblog.heuristics import *
from deepproblog.train import TrainObject
from deepproblog.utils import get_configuration, config_to_string, format_time_precise, split_dataset
from deepproblog.utils.stop_condition import Threshold, StopOnPlateau

def main(
  i = 0,
  calibrate = False,
  logging = False,
  save_model_state = True,
  model_state_name = None,
  train_with_label_noise = False,
  label_noise_probability = 0.2,
):
  dsets = ["sys_gen_{}".format(i) for i in range(3)] + ["noise_{}".format(i) for i in range(4)]
  configurations = {"method": ["gm"], "dataset": dsets, "run": range(5)}
  configuration = get_configuration(configurations, i)
  name = "clutrr_" + config_to_string(configuration) + "_" + format_time_precise()
  print(name)
  torch.manual_seed(configuration["run"])

  clutrr = CLUTRR(configuration["dataset"])
  dataset = clutrr.get_dataset(".*train", gender = True, type = "split")
  train_dataset, val_dataset = split_dataset(dataset)
  if train_with_label_noise:
    label_noise = lambda _, q: q.replace_output([random.choice([
      Constant("child"),
      Constant("child_in_law"),
      Constant("parent"),
      Constant("parent_in_law"),
      Constant("sibling"),
      Constant("sibling_in_law"),
      Constant("grandparent"),
      Constant("grandchild"),
      Constant("nephew"),
      Constant("uncle"),
      Constant("so")
    ])])
    train_dataset = MutatingDatasetWithItems(train_dataset, NoiseMutatorDecorator(label_noise_probability, label_noise))
  test_datasets = clutrr.get_dataset(".*test", gender = True, type = "split", separate = True)
  print(dataset_names[configuration["dataset"]])
  raw_datasets = {
    "rel_extract": RawCLUTRRRelExtractValidationDataset(),
    "gender_net": RawCLUTRRGenderNetValidationDataset()
  }
  loader = DataLoader(train_dataset, 4)
  rel_net_val_loader = TorchDataLoader(raw_datasets["rel_extract"], 4, collate_fn = rel_extract_dataloader_collate_fn)
  gender_net_val_loader = TorchDataLoader(raw_datasets["gender_net"], 4, collate_fn = gender_net_dataloader_collate_fn)

  embed_size = 32
  lstm = Encoder(clutrr.get_vocabulary(), embed_size, p_drop = 0.0)
  networks_evolution_collectors = {}
  lstm_net = Network(
    lstm, "encoder", optimizer = torch.optim.Adam(lstm.parameters(), lr = 1e-2)
  )
  if calibrate == True:
    rel_net = TemperatureScalingNetwork(RelNet(embed_size, 2 * embed_size), "rel_extract", rel_net_val_loader)
    gender_net = GenderNet(clutrr.get_vocabulary(), embed_size)
    gender_net = TemperatureScalingNetwork(
      gender_net,
      "gender_net",
      gender_net_val_loader,
      optimizer = torch.optim.Adam(gender_net.parameters(), lr = 1e-2),
    )
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector(
      {
        "rel_extract": rel_net_val_loader,
        "gender_net": gender_net_val_loader
      }
    )
  else:   
    rel_net = Network(RelNet(embed_size, 2 * embed_size), "rel_extract")
    gender_net = GenderNet(clutrr.get_vocabulary(), embed_size)
    gender_net = Network(
      gender_net,
      "gender_net",
      optimizer = torch.optim.Adam(gender_net.parameters(), lr = 1e-2),
    )
  rel_net.optimizer = torch.optim.Adam(rel_net.parameters(), lr = 1e-2)

  model_filename = "model_forward.pl"
  model = Model(model_filename, [rel_net, lstm_net, gender_net])

  heuristic = GeometricMean()
  if configuration["method"] == "exact":
    raise Exception('The CLUTRR experiment is currently not supported in the Exact Engine')
    # model.set_engine(ExactEngine(model))
  elif configuration["method"] == "gm":
    model.set_engine(ApproximateEngine(model, 1, heuristic, exploration = True))

  train_log = TrainObject(model, networks_evolution_collectors)
  train_log.train(
    loader,
    Threshold("Accuracy", 1.0) + StopOnPlateau("Accuracy", patience = 5, warm_up = 10),
    networks_evolution_collectors,
    initial_test = False,
    test = lambda x: [
      (
        "Accuracy",
        get_confusion_matrix(x, val_dataset, verbose = 0).accuracy(),
      )
    ],
    log_iter = 50,
    test_iter = 250,
  )

  raw_datasets["rel_extract"].update_embeddings(lstm)
  ECEs_final_calibration = {
    "rel_extract": {},
    "gender_net": {}
  }
  if calibrate:
    ECEs_final_calibration["rel_extract"]["before"] = rel_net.get_expected_calibration_error(rel_net_val_loader)
    ECEs_final_calibration["gender_net"]["before"] = gender_net.get_expected_calibration_error(gender_net_val_loader)
    rel_net.calibrate()
    gender_net.calibrate()
    ECEs_final_calibration["rel_extract"]["after"] = rel_net.get_expected_calibration_error(rel_net_val_loader)
    ECEs_final_calibration["gender_net"]["after"] = gender_net.get_expected_calibration_error(gender_net_val_loader)

  cms = []
  for dataset in test_datasets:
    cm = get_confusion_matrix(model, test_datasets[dataset], verbose = 0)
    final_acc = cm.accuracy()
    if logging == True:
      train_log.logger.comment("{}\t{}".format(dataset, final_acc))
    cms.append(cm)

  if logging == True:
    train_log.logger.comment(dumps(model.get_hyperparameters()))
    train_log.write_to_file("log/" + name)

  if save_model_state:
    if model_state_name:
      model.save_state(f"models/{model_state_name}.pth")
    else:
      model.save_state(f"models/{name}.pth")

  return [train_log, cms, ECEs_final_calibration]

if __name__ == "__main__":
  fire.Fire(main)