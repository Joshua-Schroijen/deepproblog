import fire
import subprocess
import sys

FIVE_MINUTES = 300

class FixedSizeList:
  def __init__(self, n):
    self.the_list = []
    self.n = n

  def append(self, e):
    if (len(self.the_list) < self.n):
      self.the_list.append(e)
    else:
      self.the_list = self.the_list[1:]
      self.the_list.append(e)

  def __iter__(self):
    self.iter_i = 0
    return self

  def __next__(self):
    if self.iter_i < len(self.the_list):
      return self.the_list[self.iter_i]
    else:
      raise StopIteration

def main(n = 10):
  result = subprocess.run(['python3', '-u', '-m', 'trace', '--trace'], capture_output = True, text = True)
  output = result.stdout.splitlines()

  input_fsl = FixedSizeList(n)
  for line in output:
    input_fsl.append(line)

  for line in input_fsl:
    print(line)

if __name__ == "__main__":
  fire.Fire(main)