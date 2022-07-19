# System requirements
DeepProbLog is memory intensive. 32+ GB of RAM is not required, but is not a luxury. Make sure ample swap space is available.
DeepProbLog requires a Linux-environment, be it a containerized or native one. Running Ubuntu 22.04+ or an equivalent distribution with Python 3.10+ tools is recommendable.

# Installation using setup.sh
To install a DeepProbLog virtual environment on your system along with the necessary external dependencies, you can execute setup.sh
Run `chmod +x setup.sh` followed by `./setup.sh`. You will be asked for a root password, since the script uses sudo. Please make sure to run the script as a user in the sudo group. Finally, run `source DevEnv/bin/activate` to activate the new virtual environment.

# Manual installation
To develop in this codebase, install it in your Python installation as a local package in editable mode  
`pip3 install -e .` (in the directory containing this very file)  

If on Ubuntu, now make sure you have all GCC/G++ tooling to build PySDD and PySwip:  
`sudo apt install gcc libpython3-dev`  
  
Also make sure to have SWI-Prolog installed  
`sudo apt install swi-prolog`  

Uninstall PySwip and PySDD and reinstall them using these commands:  
`pip3 install git+https://github.com/wannesm/PySDD.git#egg=PySDD`  
`pip3 install git+https://github.com/ML-KULeuven/pyswip@development#egg=pyswip`
