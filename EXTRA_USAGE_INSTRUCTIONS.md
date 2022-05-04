To develop in this codebase, install it in your Python installation as a local package in editable mode  
`pip3 install -e .` (in the directory containing this very file)  

If on Ubuntu, now make sure you have all GCC/G++ tooling to build PySDD and PySwip:  
`sudo apt install gcc libpython3-dev`  
  
Also make sure to have SWI-Prolog installed  
`sudo apt install swi-prolog`  

Uninstall PySwip and PySDD and reinstall them using these commands:  
`pip3 install git+https://github.com/wannesm/PySDD.git#egg=PySDD`  
`pip3 install ggit+https://github.com/ML-KULeuven/pyswip@development#egg=pyswip`
