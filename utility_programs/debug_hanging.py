import fire
import subprocess

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
      nxt = self.the_list[self.iter_i]
      self.iter_i += 1
      return nxt
    else:
      raise StopIteration

def main(i, n = 10, t = FIVE_MINUTES):
  proc = subprocess.Popen(['python3', '-u', '-m', 'trace', '--trace', i], stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)
  try:
    outs, _ = proc.communicate(timeout = t)
  except subprocess.TimeoutExpired:
    proc.kill()
    outs, _ = proc.communicate()

  output = outs.splitlines()

  input_fsl = FixedSizeList(n)
  for line in output:
    input_fsl.append(line)
 
  for line in input_fsl:
    print(line)

if __name__ == "__main__":
  fire.Fire(main)