import ast
import fire
from PyInquirer import prompt

def present_sample_get_label(sample):
  return prompt([
    {
      'type': 'input',
      'name': 'label',
      'message': sample,
    }
  ])["label"]

def confirm_label():
  return prompt([
    {
        'type': 'confirm',
        'message': 'Label correct?',
        'name': 'confirmed',
        'default': False,
    }
  ])["confirmed"]

def main(infile_name, outfile_name):
  with open(infile_name, "r") as infile, open(outfile_name, "w") as outfile:
    for inline in infile:
      sample = ast.literal_eval(inline.strip())
      while True:
        label = present_sample_get_label(str(sample))
        if confirm_label():
          sample.append(label)
          outfile.write(str(sample) + "\n")
          break

if __name__ == "__main__":
  fire.Fire(main)