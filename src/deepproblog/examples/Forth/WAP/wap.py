import fire

from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Forth.WAP.wap_network import get_networks
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.train import train_model

def main(
  calibrate = False,
  calibrate_after_each_train_iteration = False
):
  train_queries = QueryDataset("data/train.pl")
  dev_queries = QueryDataset("data/dev.pl")
  test_queries = QueryDataset("data/test.pl")

  networks = get_networks(0.005, 0.5)

  networks_evolution_collectors = {}
  if calibrate == True:
    train_networks = [TemperatureScalingNetwork(x[0], x[1], DataLoader(dev_queries, 10), x[2]) for x in networks]
    test_networks = \
      [TemperatureScalingNetwork(networks[0][0], networks[0][1], DataLoader(dev_queries, 10))] + \
      [TemperatureScalingNetwork(x[0], x[1], DataLoader(dev_queries, 10), k = 1) for x in networks[1:]]
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector()
  else:
    train_networks = [Network(x[0], x[1], x[2]) for x in networks]
    test_networks = [Network(networks[0][0], networks[0][1])] + [
      Network(x[0], x[1], k=1) for x in networks[1:]
    ]

  model = Model("wap.pl", train_networks)
  model.set_engine(ExactEngine(model), cache = False)
  test_model = Model("wap.pl", test_networks)
  test_model.set_engine(ExactEngine(test_model), cache = False)

  train_obj = train_model(
    model,
    DataLoader(train_queries, 10),
    40,
    networks_evolution_collectors,
    log_iter = 10,
    test = lambda x: [
        ("Accuracy", get_confusion_matrix(test_model, test_queries).accuracy())
    ],
    test_iter=30,
  )
  return [train_obj, get_confusion_matrix(test_model, test_queries, verbose = 0)]

if __name__ == "__main__":
  fire.Fire(main())
