import json
import logging
import os
from ssl import SSL_ERROR_SYSCALL

import fire
import numpy as np
import matplotlib.pyplot as plt

import deepproblog.examples.MNIST.addition as addition
import deepproblog.examples.MNIST.addition_mil as addition_mil
import deepproblog.examples.MNIST.addition_noisy as addition_noisy

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
RESULTS_DIR = os.path.join(os.getcwd(), "calibration_evaluator_results")

def log_heading(logger, txt, length = 50):
  txt_len_plus_1 = len(txt) + 1
  logger.info(txt + " " + ("-" * (length - txt_len_plus_1)))

def log_subheading(logger, txt, length = 50):
  txt_len_plus_3 = len(txt) + 3
  logger.info("- " + txt + " " + ("-" * (length - txt_len_plus_3)))

def log_empty_line(logger):
  logger.info("")

def plot_loss_curve(loss_history, name, title):
  plt.figure()
  plt.plot(loss_history)
  plt.title(title)
  plt.xticks(np.arange(0, len(loss_history), round(len(loss_history) / 10)))
  plt.savefig(os.path.join(RESULTS_DIR, name))

def dump_data_of_interest(filename, train_object, confusion_matrix):
  data_of_interest = {
    "loss_history": train_object.loss_history,
	"accuracy": confusion_matrix.accuracy()
  }
  with open(os.path.join(RESULTS_DIR, filename), "w") as f:
    json.dump(data_of_interest, f, indent = 6)

def main(logfile="calibration_evaluation.txt"):
  if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
  logging.basicConfig(filename=logfile, level=logging.INFO, format=LOG_FORMAT, filemode='w')
  logger = logging.getLogger()

  initial_working_directory = os.getcwd()

  os.chdir("./MNIST")

  log_heading(logger, "Evaluating addition")

  log_subheading(logger, "Without calibration")
  [train, confusion_matrix] = addition.main(calibrate = False, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_experiment_1.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "Without calibration after each train iteration")
  [train, confusion_matrix] = addition.main(calibrate = True, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_experiment_2.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "With calibration after each train iteration")
  [train, confusion_matrix] = addition.main(calibrate = True, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_experiment_3.json", train, confusion_matrix)
  log_empty_line(logger)

  os.chdir(initial_working_directory)

if __name__ == '__main__':
  fire.Fire(main)