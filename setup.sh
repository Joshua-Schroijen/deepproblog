#!/bin/bash

sudo apt-get -y update && sudo apt-get -y upgrade
sudo apt-get install -y python3-pip python3.9-venv python3.10-venv gcc swi-prolog libpython3-dev unzip
python3 -m venv DevEnv
source DevEnv/bin/activate
pip3 install numpy matplotlib fire gdown
pip3 install -e .
pip3 uninstall -y pyswip pysdd
pip3 install git+https://github.com/wannesm/PySDD.git#egg=PySDD
pip3 install git+https://github.com/yuce/pyswip@master#egg=pyswip

bash src/deepproblog/examples/HWF/data/download_hwf.sh