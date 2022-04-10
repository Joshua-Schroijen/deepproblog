#!/bin/bash

if dpkg -l | grep -q gcc; then
  sudo apt-get install gcc
fi
if dpkg -l | grep -q swi-prolog; then
  sudo apt-get install swi-prolog
fi
python3 -m venv DevEnv
source DevEnv/bin/activate
pip3 install -e .
pip3 uninstall pyswip pysdd
pip3 install git+https://github.com/wannesm/PySDD.git#egg=PySDD
pip3 install git+https://github.com/yuce/pyswip@master#egg=pyswip