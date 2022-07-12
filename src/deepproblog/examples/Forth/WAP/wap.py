import fire
from torch.utils.data import DataLoader as TorchDataLoader

from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Forth.WAP.data.for_calibration import RawWAPOp1ValidationDataset, RawWAPOp2ValidationDataset, RawWAPPermuteValidationDataset, RawWAPSwapValidationDataset, op1_dataloader_collate_fn, op2_dataloader_collate_fn, permute_dataloader_collate_fn, swap_dataloader_collate_fn
from deepproblog.examples.Forth.WAP.wap_network import get_networks
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.train import train_model

def main(
  calibrate = False
):
  train_queries = QueryDataset("data/train_s.pl")
  dev_queries = QueryDataset("data/dev_s.pl")
  test_queries = QueryDataset("data/test_s.pl")
  raw_datasets = {
    "nn_permute": RawWAPPermuteValidationDataset(),
    "nn_op1": RawWAPOp1ValidationDataset(),
    "nn_swap": RawWAPSwapValidationDataset(),
    "nn_op2": RawWAPOp2ValidationDataset()
  }
  collate_fns = {
    "nn_permute": permute_dataloader_collate_fn,
    "nn_op1":  op1_dataloader_collate_fn,
    "nn_swap": swap_dataloader_collate_fn,
    "nn_op2": op2_dataloader_collate_fn,
  }

  networks = get_networks(0.005, 0.5)

  networks_evolution_collectors = {}
  if calibrate == True:
    train_networks = \
      [Network(networks[0][0], networks[0][1], networks[0][2])] + \
      [TemperatureScalingNetwork(x[0], x[1], TorchDataLoader(raw_datasets[x[1]], 10, collate_fn = collate_fns[x[1]]), x[2]) for x in networks[1:]]
    test_networks = \
      [Network(networks[0][0], networks[0][1])] + \
      [TemperatureScalingNetwork(x[0], x[1], TorchDataLoader(raw_datasets[x[1]], 10, collate_fn = collate_fns[x[1]]), k = 1) for x in networks[1:]]
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

  rnn = networks[0][0]
  for raw_dataset in raw_datasets.values():
    raw_dataset.update_embeddings(rnn)
  if calibrate:
    for train_network in train_networks[1:]:
      train_network.calibrate()

  #return [train_obj, get_confusion_matrix(test_model, test_queries, verbose = 0)]
  return [train_obj, get_confusion_matrix(model, test_queries, verbose = 0)]

if __name__ == "__main__":
  fire.Fire(main)
