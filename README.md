# Transferable Dialogue Systems and User Simulators

NB: The codebase is not completely clean and still in the process of refactoring. This would not affect the produced results.

## Requirements
python3.6 and the packages in `requirements.txt`. Install them via virtual environment:
```console
>>> python -m venv env
>>> source env/bin/activate
>>> pip install -r requirements.txt
```

## Full Training Experiments
Pre-process the raw MultiWOZ data
```console
>>> bash src/prepare_data.sh
```

Training from scratch using Supervised Learning
```console
>>> bash src/train-SL.sh [your_sl_model_name]
```

### Continued training using Reinforcement Learning
```console
>>> bash src/train-RL.sh [your_sl_model_name] [your_rl_model_name]
```

### Testing your model
Simply run...
```console
>>> bash src/xxx.sh
```
- mode: "test"

### Test the provided model
We provide a trained model to reproduce the benchmark result reported in the paper (Table2).

First, unzip the model file by
```console
>>> cd checkpoint/
>>> tar zxvf provided_joust.tar.gz
>> cd ..
```

Then run the testing script with the model name `provided_joust`
```console
>>> bash src/test.sh provided_joust
```
Again, the results produced by interacting against the test set and the user simulator 
will be stored in the folders `corpus_interact_output/` and `user_interact_output/` respectively.

Note that the produced results by this model is slightly different to the
reported numbers (e.g., 95.6 vs 96.0) as they are averaged by three runs.

### Create transfer learning data
```console
>>> bash src/prepare_data_transfer_learning.sh
```
Data for single-to-multi domain setup will be created in `data/single_to_multi/`.
Data for domain adaptation setup will be created in `data/domain_transfer/`.



TODO
- Add paper link
- make sure benchmark eval numbers can be reproduced
- clean code, argument.py at the end
  (no need to train from scratch)
- separate run bash, (train_SL, train_RL, test.sh)
- provide one best benchmark model (compressed) for both two setups 

