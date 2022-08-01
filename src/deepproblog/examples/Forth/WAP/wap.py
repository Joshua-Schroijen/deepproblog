import fire
import random
from torch.utils.data import DataLoader as TorchDataLoader

from problog.logic import Constant

from deepproblog.calibrated_network import TemperatureScalingNetwork, NetworkECECollector
from deepproblog.dataset import DataLoader, QueryDataset, NoiseMutatorDecorator, MutatingDatasetWithItems
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Forth.WAP.data.for_calibration import RawWAPOp1ValidationDataset, RawWAPOp2ValidationDataset, RawWAPPermuteValidationDataset, RawWAPSwapValidationDataset, op1_dataloader_collate_fn, op2_dataloader_collate_fn, permute_dataloader_collate_fn, swap_dataloader_collate_fn
from deepproblog.examples.Forth.WAP.wap_network import get_networks
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.query import Query
from deepproblog.train import train_model

def main(
  calibrate = False,
  save_model_state = True,
  model_state_name = None,
  train_with_label_noise = False,
  label_noise_probability = 0.2,
):
  train_queries = QueryDataset("data/train_s.pl")
  dev_queries = QueryDataset("data/dev_s.pl")
  test_queries = QueryDataset("data/test_s.pl")
  if train_with_label_noise:
    def label_noise(_, q: Query):
      flip = random.choice([False, True])
      if flip:
        q.replace_output([Constant(float(random.randint(1, 86)))])
      return q
    train_queries = MutatingDatasetWithItems(train_queries, NoiseMutatorDecorator(label_noise_probability, label_noise))
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
  raw_validation_dataloaders = {
    k: TorchDataLoader(raw_datasets[k], 10, collate_fn = collate_fns[k]) for k in raw_datasets.keys()
  }

  networks = get_networks(0.005, 0.5)
  networks_evolution_collectors = {}
  if calibrate == True:
    train_networks = \
      [Network(networks[0][0], networks[0][1], networks[0][2])] + \
      [TemperatureScalingNetwork(x[0], x[1], raw_validation_dataloaders[x[1]], x[2]) for x in networks[1:]]
    test_networks = \
      [Network(networks[0][0], networks[0][1])] + \
      [TemperatureScalingNetwork(x[0], x[1], raw_validation_dataloaders[x[1]], k = 1) for x in networks[1:]]
    networks_evolution_collectors["calibration_collector"] = NetworkECECollector(
      {
        n.name: raw_validation_dataloaders[n.name] for n in networks[1:]
      }
    )
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
      ("Accuracy", get_confusion_matrix(test_model, dev_queries, verbose = 0).accuracy())
    ],
    test_iter=30,
  )

  rnn = networks[0][0]
  for raw_dataset in raw_datasets.values():
    raw_dataset.update_embeddings(rnn)
  ECEs_final_calibration = {
    n.name: {} for n in train_networks[1:]
  }
  if calibrate:
    for train_network in train_networks[1:]:
      ECEs_final_calibration[train_network.name]["before"] = train_network.get_expected_calibration_error(raw_validation_dataloaders[train_network.name])
      train_network.calibrate()
      ECEs_final_calibration[train_network.name]["after"] = train_network.get_expected_calibration_error(raw_validation_dataloaders[train_network.name])

  if save_model_state:
    if model_state_name:
      model.save_state(f"snapshot/{model_state_name}.pth")
    else:
      model.save_state(f"snapshot/forth_WAP.pth")

  return [train_obj, get_confusion_matrix(test_model, test_queries, verbose = 0), ECEs_final_calibration]

if __name__ == "__main__":
  fire.Fire(main)
