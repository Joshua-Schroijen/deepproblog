import os

import numpy as np
import matplotlib.pyplot as plt

import deepproblog.examples.MNIST.addition as addition
import deepproblog.examples.MNIST.addition_mil as addition_mil
import deepproblog.examples.MNIST.addition_noisy as addition_noisy

RESULTS_DIR = os.path.join(os.getcwd(), "calibration_evaluator_results")

def plot_loss_curve(loss_history, name, title):
  plt.figure()
  plt.plot(loss_history)
  plt.title(title)
  plt.xticks(np.arange(0, len(loss_history), round(len(loss_history) / 10)))
  plt.savefig(os.path.join(RESULTS_DIR, name))

if __name__ == '__main__':
  if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

  initial_working_directory = os.getcwd()

  os.chdir("./MNIST")

  print("RUNNING ADDITION " + ("-" * 33))
  print("- WITHOUT CALIBRATION " + ("-" * 28))
  [train, _] = addition.main(calibrate=False)
  plot_loss_curve(train.loss_history, "addition_wo_calibration", "Addition without calibration")
  print("- WITH CALIBRATION " + ("-" * 31))
  [train, _] = addition.main(calibrate=True)
  plot_loss_curve(train.loss_history, "addition_w_calibration", "Addition with calibration")

  # print("RUNNING ADDITION_MIL " + ("-" * 29))
  # addition_mil.main()
  # print("RUNNING ADDITION_NOISY " + ("-" * 29))
  # addition_noisy.main()

  os.chdir(initial_working_directory)
