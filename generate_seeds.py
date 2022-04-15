import fire
import os

def generate_seeds(no_seeds):
  for _ in range(no_seeds):
    print(int.from_bytes(os.urandom(32), 'big'))

if __name__ == "__main__":
  fire.Fire(generate_seeds)