#!/bin/bash

sudo apt-get -y update
if dpkg -l | grep -q gcc; then
  sudo apt-get install -y gcc
fi
if dpkg -l | grep -q swi-prolog; then
  sudo apt-get install -y swi-prolog
fi
if dpkg -l | grep -q libpython3-dev; then
  sudo apt-get install -y libpython3-dev
fi
python3 -m venv DevEnv
source DevEnv/bin/activate
pip3 install numpy matplotlib fire
pip3 install -e .
pip3 uninstall -y pyswip pysdd
pip3 install git+https://github.com/wannesm/PySDD.git#egg=PySDD
pip3 install git+https://github.com/yuce/pyswip@master#egg=pyswip