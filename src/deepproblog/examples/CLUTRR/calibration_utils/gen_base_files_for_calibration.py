import sys
custom_solver_module = __import__('solver')
sys.modules['.solver'] = custom_solver_module
sys.modules['deepproblog.solver'] = custom_solver_module

import fire
from torch.utils.data import DataLoader as TorchDataLoader
from json import dumps

from deepproblog.engines import ApproximateEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.network import Network
from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.model import Model
from deepproblog.dataset import DataLoader
from deepproblog.examples.CLUTRR.calibration_utils.architecture import Encoder, RelNet, GenderNet
from deepproblog.examples.CLUTRR.calibration_utils.data import CLUTRR, dataset_names
from deepproblog.heuristics import *
from deepproblog.train import TrainObject
from deepproblog.utils import get_configuration, config_to_string, format_time_precise, split_dataset
from deepproblog.utils.stop_condition import Threshold, StopOnPlateau

def main(
  i = 0
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
  print(dataset_names[configuration["dataset"]])
  loader = DataLoader(val_dataset, 4)

  embed_size = 32
  lstm = Encoder(clutrr.get_vocabulary(), embed_size, p_drop = 0.0)

  networks_evolution_collectors = {}
  lstm_net = Network(
    lstm, "encoder", optimizer = torch.optim.Adam(lstm.parameters(), lr = 1e-2)
  )
  rel_net = Network(RelNet(embed_size, 2 * embed_size), "rel_extract")
  gender_net = GenderNet(clutrr.get_vocabulary(), embed_size)
  gender_net = Network(
    gender_net,
    "gender_net",
    optimizer = torch.optim.Adam(gender_net.parameters(), lr=1e-2),
  )
  rel_net.optimizer = torch.optim.Adam(rel_net.parameters(), lr = 1e-2)

  model_filename = "model_forward.pl"
  model = Model(model_filename, [rel_net, lstm_net, gender_net])

  heuristic = GeometricMean()
  if configuration["method"] == "exact":
    raise Exception('The CLUTRR experiment is currently not supported in the Exact Engine')
  elif configuration["method"] == "gm":
    model.set_engine(ApproximateEngine(model, 1, heuristic, exploration = True))

  train_log = TrainObject(model)
  train_log.train(
    loader,
    Threshold("Accuracy", 1.0) + StopOnPlateau("Accuracy", patience = 5, warm_up = 10),
    initial_test = False,
    test = lambda x: [
      (
        "Accuracy",
        get_confusion_matrix(x, val_dataset, verbose = 0).accuracy(),
      )
    ],
    log_iter = 50,
  )

if __name__ == "__main__":
  fire.Fire(main)
