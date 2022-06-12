import fire
import sys
from json import dumps

from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.network import Network
from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.model import Model
from deepproblog.dataset import DataLoader
from deepproblog.examples.CLUTRR.architecture import Encoder, RelNet, GenderNet
from deepproblog.examples.CLUTRR.data import CLUTRR, dataset_names
from deepproblog.heuristics import *
from deepproblog.train import TrainObject
from deepproblog.utils import get_configuration, config_to_string, format_time_precise, split_dataset
from deepproblog.utils.stop_condition import Threshold, StopOnPlateau

def main(
  i = 0,
  calibrate = False,
  calibrate_after_each_train_iteration = False,
  logging = False
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
  test_datasets = clutrr.get_dataset(".*test", gender = True, type = "split", separate = True)
  print(dataset_names[configuration["dataset"]])
  loader = DataLoader(train_dataset, 4)
  val_loader = DataLoader(val_dataset, 4)

  embed_size = 32
  lstm = Encoder(clutrr.get_vocabulary(), embed_size, p_drop = 0.0)

  networks_evolution_collectors = {}
  if calibrate == True:
    lstm_net = TemperatureScalingNetwork(
      lstm, "encoder", val_loader, optimizer = torch.optim.Adam(lstm.parameters(), lr = 1e-2)
    )
    rel_net = Network(RelNet(embed_size, 2 * embed_size), "rel_extract", val_loader)
    gender_net = GenderNet(clutrr.get_vocabulary(), embed_size)
    gender_net = Network(
      gender_net,
      "gender_net",
      val_loader,
      optimizer = torch.optim.Adam(gender_net.parameters(), lr = 1e-2),
    )
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector()
  else:
    lstm_net = Network(
      lstm, "encoder", optimizer = torch.optim.Adam(lstm.parameters(), lr = 1e-2)
    )    
    rel_net = Network(RelNet(embed_size, 2 * embed_size), "rel_extract")
    gender_net = GenderNet(clutrr.get_vocabulary(), embed_size)
    gender_net = Network(
      gender_net,
      "gender_net",
      optimizer=torch.optim.Adam(gender_net.parameters(), lr=1e-2),
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

  train_log = TrainObject(model)
  train_log.train(
    loader,
    Threshold("Accuracy", 1.0) + StopOnPlateau("Accuracy", patience=5, warm_up = 10),
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

  model.save_state("models/" + name + ".pth")

  for dataset in test_datasets:
    cm = get_confusion_matrix(model, test_datasets[dataset], verbose = 0)
    final_acc = cm.accuracy()
    if logging == True:
      train_log.logger.comment("{}\t{}".format(dataset, final_acc))

  if logging == True:
    train_log.logger.comment(dumps(model.get_hyperparameters()))
    train_log.write_to_file("log/" + name)

if __name__ == "__main__":
  fire.Fire(main)