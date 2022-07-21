#!/bin/bash

cat << EOF >> ~/.profile
if [ -d "$PWD/utility_programs" ] ; then
    PYTHONPATH="$PWD/utility_programs:\$PYTHONPATH"
fi
EOF

sudo apt-get -y update && sudo apt-get -y upgrade
sudo apt-get install -y python3-pip python3.10-venv gcc swi-prolog libpython3-dev unzip
python3 -m venv DevEnv
source DevEnv/bin/activate
pip3 install fire gdown matplotlib numpy problog torch
pip3 install -e .
pip3 uninstall -y pyswip pysdd
pip3 install git+https://github.com/wannesm/PySDD.git#egg=PySDD
pip3 install git+https://github.com/ML-KULeuven/pyswip@development#egg=pyswip

bash src/deepproblog/examples/HWF/data/download_hwf.sh
bash src/deepproblog/examples/Coins/data/download_image_data.sh
bash src/deepproblog/examples/Poker/data/download_images.sh
    