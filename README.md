# Transferable Dialogue Systems and User Simulators

## Requirements
python3.6 and the packages in `requirements.txt`. Install them via virtual environment:
```console
>>> python -m venv env
>>> source env/bin/activate
>>> pip install -r requirements.txt
```

## Preparation
Create necessary directories by
```console
>>> bash src/prepare_directory.sh
```

Create full training data
```console
>>> bash src/prepare_data.sh
```


TODO
- make sure benchmark eval numbers can be reproduced
- clean code, argument.py at the end
  (no need to train from scratch)
- separate run bash, (train_SL, train_RL, test.sh)
- provide one best benchmark model (compressed) for both two setups 

