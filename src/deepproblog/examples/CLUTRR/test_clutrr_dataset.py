from deepproblog.examples.CLUTRR.data import CLUTRR
from deepproblog.dataset import DataLoader
from deepproblog.utils import get_configuration

if __name__ == "__main__":
  i = 0
  dsets = ["sys_gen_{}".format(i) for i in range(3)] + ["noise_{}".format(i) for i in range(4)]

  configurations = {"method": ["gm"], "dataset": dsets, "run": range(5)}
  configuration = get_configuration(configurations, i)

  clutrr = CLUTRR(configuration["dataset"])
  dataset = clutrr.get_dataset(".*train", gender = True, type = "split")
  loader = DataLoader(dataset, 4)
  for batch in loader:
    breakpoint()