# Transferable Dialogue Systems and User Simulators

NB: The codebase is not completely clean and still in the process of refactoring. This would not affect the produced results.

## Requirements
python3.6 and the packages in `requirements.txt`. Install them via virtual environment:
```console
>>> python -m venv env
>>> source env/bin/activate
>>> pip install -r requirements.txt
```

Create necessary directories by
```console
>>> bash src/prepare_directory.sh
```

##Full Training Experiments
###Pre-process the raw MultiWOZ data first
```console
>>> bash src/prepare_data.sh
```

###Training from scratch using Supervised Learning
Simply run ... the key hyper-parameters are listed below
```console
>>> bash src/...sh 
```
Training hyper-parameters
- mode: "pretrain"
- mode:


###Continued training using Reinforcement Learning
Simply run...
```console
>>> bash src/xxx.sh
```
- mode: "rl"

###Testing the model
Simply run...
```console
>>> bash src/xxx.sh
```
- mode: "test"





TODO
- make sure benchmark eval numbers can be reproduced
- clean code, argument.py at the end
  (no need to train from scratch)
- separate run bash, (train_SL, train_RL, test.sh)
- provide one best benchmark model (compressed) for both two setups 

