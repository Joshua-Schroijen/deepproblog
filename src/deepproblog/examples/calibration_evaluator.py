import json
import logging
import logging.config
import os

import fire
import numpy as np
import matplotlib.pyplot as plt

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
RESULTS_DIR = os.path.join(os.getcwd(), "calibration_evaluator_results")

MNIST_ADDITION_SEED = 66815121350953911695398616902400003414994440380511384177123061993195380927844
MNIST_ADDITION_NOISY_SEED =65383380116126662821362947724785949135007927266136665876324451069998367473092
FORTH_SEED = 112195398336644232901763685921564964554228813564928553493693669635703169635702
HWF_SEED = 30575021600727355504011411996420506533046860355500394129121043613226662678680
POKER_SEED = 15384679245914431582552102544026922064770126905211487330762948489981255464981
COINS_SEED = 55584395892288086672466369677165805286392349207122007596909015435442832056023
CLUTRR_SEED = 78171602626066457765912268658639150623375646781597908325888418343003898255803

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

def evaluate_MNIST_addition(logger):
  os.chdir("MNIST")
  import deepproblog.examples.MNIST.addition as addition

  log_heading(logger, "Evaluating MNIST addition")

  log_subheading(logger, "Without calibration")
  [train, confusion_matrix] = addition.main(calibrate = False, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_addition_ff.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "Without calibration after each train iteration")
  [train, confusion_matrix] = addition.main(calibrate = True, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_addition_tf.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "With calibration after each train iteration")
  [train, confusion_matrix] = addition.main(calibrate = True, calibrate_after_each_train_iteration = True)
  dump_data_of_interest("calibration_evaluation_addition_tt.json", train, confusion_matrix)
  log_empty_line(logger)

  os.chdir("..")

def evaluate_MNIST_noisy(logger):
  os.chdir("MNIST")
  import deepproblog.examples.MNIST.addition_noisy as addition_noisy

  log_heading(logger, "Evaluating MNIST noisy addition")

  log_subheading(logger, "Without calibration")
  [train, confusion_matrix] = addition_noisy.main(calibrate = False, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_addition_noisy_addition_ff.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "Without calibration after each train iteration")
  [train, confusion_matrix] = addition_noisy.main(calibrate = True, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_addition_noisy_addition_tf.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "With calibration after each train iteration")
  [train, confusion_matrix] = addition_noisy.main(calibrate = True, calibrate_after_each_train_iteration = True)
  dump_data_of_interest("calibration_evaluation_addition_noisy_addition_tt.jsonn", train, confusion_matrix)
  log_empty_line(logger)

  os.chdir("..")

def evaluate_coins(logger):
  os.chdir("Coins")
  import deepproblog.examples.Coins.coins as coins

  log_heading(logger, "Evaluating Coins")

  log_subheading(logger, "Without calibration")
  [train, confusion_matrix] = coins.main(calibrate = False, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_coins_ff.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "Without calibration after each train iteration")
  [train, confusion_matrix] = coins.main(calibrate = True, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_coins_tf.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "With calibration after each train iteration")
  [train, confusion_matrix] = coins.main(calibrate = True, calibrate_after_each_train_iteration = True)
  dump_data_of_interest("calibration_evaluation_coins_tt.json", train, confusion_matrix)
  log_empty_line(logger)

  os.chdir("..")

def evaluate_poker(logger):
  os.chdir("Poker")
  import deepproblog.examples.Poker.poker as poker

  log_heading(logger, "Evaluating Poker")

  log_subheading(logger, "Without calibration")
  [train, confusion_matrix] = poker.main(calibrate = False, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_poker_ff.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "Without calibration after each train iteration")
  [train, confusion_matrix] = poker.main(calibrate = True, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_poker_tf.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "With calibration after each train iteration")
  [train, confusion_matrix] = poker.main(calibrate = True, calibrate_after_each_train_iteration = True)
  dump_data_of_interest("calibration_evaluation_poker_tt.json", train, confusion_matrix)
  log_empty_line(logger)

  os.chdir("..")

def evaluate_HWF(logger):
  os.chdir("HWF")
  import deepproblog.examples.HWF.hwf as hwf

  log_heading(logger, "Evaluating HWF")

  log_subheading(logger, "Without calibration")
  [train, confusion_matrix] = hwf.main(calibrate = False, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_hwf_ff.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "Without calibration after each train iteration")
  [train, confusion_matrix] = hwf.main(calibrate = True, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_hwf_tf.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "With calibration after each train iteration")
  [train, confusion_matrix] = hwf.main(calibrate = True, calibrate_after_each_train_iteration = True)
  dump_data_of_interest("calibration_evaluation_hwf_tt.json", train, confusion_matrix)
  log_empty_line(logger)

  os.chdir("..")

def evaluate_Forth_Add(logger):
  os.chdir(os.path.join("Forth", "Add"))
  import deepproblog.examples.Forth.Add.add as forth_add

  log_heading(logger, "Evaluating Forth/Add")

  log_subheading(logger, "Without calibration")
  [train, confusion_matrix] = forth_add.main(calibrate = False, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_forth_add_ff.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "Without calibration after each train iteration")
  [train, confusion_matrix] = forth_add.main(calibrate = True, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_forth_add_tf.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "With calibration after each train iteration")
  [train, confusion_matrix] = forth_add.main(calibrate = True, calibrate_after_each_train_iteration = True)
  dump_data_of_interest("calibration_evaluation_forth_add_tt.json", train, confusion_matrix)
  log_empty_line(logger)

  os.chdir("..")

def evaluate_Forth_Sort(logger):
  os.chdir(os.path.join("Forth", "Sort"))
  import deepproblog.examples.Forth.Sort.sort as forth_sort

  log_heading(logger, "Evaluating Forth/Sort")

  log_subheading(logger, "Without calibration")
  [train, confusion_matrix] = forth_sort.main(calibrate = False, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_forth_sort_ff.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "Without calibration after each train iteration")
  [train, confusion_matrix] = forth_sort.main(calibrate = True, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_forth_sort_tf.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "With calibration after each train iteration")
  [train, confusion_matrix] = forth_sort.main(calibrate = True, calibrate_after_each_train_iteration = True)
  dump_data_of_interest("calibration_evaluation_forth_sort_tt.json", train, confusion_matrix)
  log_empty_line(logger)

  os.chdir("..")

def evaluate_Forth_WAP(logger):
  os.chdir(os.path.join("Forth", "WAP"))
  import deepproblog.examples.Forth.WAP.wap as forth_wap

  log_heading(logger, "Evaluating Forth/WAP")

  log_subheading(logger, "Without calibration")
  [train, confusion_matrix] = forth_wap.main(calibrate = False, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_forth_wap_ff.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "Without calibration after each train iteration")
  [train, confusion_matrix] = forth_wap.main(calibrate = True, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_forth_wap_tf.json", train, confusion_matrix)
  log_empty_line(logger)

  os.chdir("..")

def evaluate_CLUTRR(logger):
  os.chdir("CLUTRR")
  import deepproblog.examples.CLUTRR.clutrr as clutrr

  log_heading(logger, "Evaluating CLUTRR")

  log_subheading(logger, "Without calibration")
  [train, confusion_matrix] = clutrr.main(calibrate = False, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_clutrr_ff.json", train, confusion_matrix)
  log_empty_line(logger)

  log_subheading(logger, "With calibration")
  log_subheading(logger, "Without calibration after each train iteration")
  [train, confusion_matrix] = clutrr.main(calibrate = True, calibrate_after_each_train_iteration = False)
  dump_data_of_interest("calibration_evaluation_clutrr_tf.json", train, confusion_matrix)
  log_empty_line(logger)

  os.chdir("..")

def main(logfile="calibration_evaluation.txt"):
  if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
  logging.config.fileConfig('calibration_evaluator_logging.ini')
  logger = logging.getLogger(__name__)
  initial_working_directory = os.getcwd()

  evaluate_MNIST_addition(logger)
  evaluate_MNIST_noisy(logger)
  # evaluate_coins(logger)
  # evaluate_poker(logger)
  # evaluate_HWF(logger)
  # evaluate_Forth_Add(logger)
  # evaluate_Forth_Sort(logger)
  # evaluate_Forth_WAP(logger)
  # evaluate_CLUTRR(logger)

  os.chdir(initial_working_directory)

if __name__ == '__main__':
  fire.Fire(main)