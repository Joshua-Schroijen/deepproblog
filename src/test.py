import fire

def main(infile_name):
  with open(infile_name, "r") as f:
    for l in f:
      print(l)

if __name__ == "__main__":
  fire.Fire(main)