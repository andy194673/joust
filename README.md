# Transferable Dialogue Systems and User Simulators

[**Paper**](https://aclanthology.org/2021.acl-long.13.pdf) |
[**Full Training**](#Full-Training) | [**Transfer Learning**](#Transfer-Learning) |
[**Citation**](#Citation) | [**Contact**](#Contact-Us)

This repository contains the source code for the paper [Transferable Dialogue Systems and User Simulators](https://aclanthology.org/2021.acl-long.13.pdf),
which is accepted by ACL-IJCNLP 2021.
In this work, we propose a novel joint optimisation framework of both dialogue system and user simulator for multi-domain task-oriented 
dialogues (MultiWOZ). We further demonstrate the efficacy of our model in two transfer learning setups.
Please refer to the paper for more details.

NB: The codebase is not completely clean and still in the process of refactoring.

## Requirements
python3.6 and the packages in `requirements.txt`. Install them via virtual environment:
```console
>>> python -m venv env
>>> source env/bin/activate
>>> pip install -r requirements.txt
```

## Full Training
### Pre-process the raw MultiWOZ data
```console
>>> bash src/prepare_data.sh
```

### Training from scratch using Supervised Learning
Run the following command to train both ent-to-end agents.
```console
>>> bash src/train-SL.sh [your_sl_model_name]
```
`your_sl_model_name` can be a random string, and the resulting model will be stored at `checkpoint/your_sl_model_name/`.


### Continued training using Reinforcement Learning
Run the following command to fine-tune the model using RL through agents' interaction.
```console
>>> bash src/train-RL.sh [your_sl_model_name] [your_rl_model_name]
```
`your_rl_model_name` can be a random string, and the resulting model will be stored at `checkpoint/your_rl_model_name/`.


### Testing the model
Through the following command, the trained dialogue agent will be evaluated through interacting with the test corpus
and the user simulator.
```console
>>> bash src/test.sh [your_model_name]
```
Three output files will be produced for each type of interaction:
- DST: predictions of belief state
- Policy: predictions of dialogue act
- NLG: predictions of natural language response

Results of interacting against test corpus will be saved in `corpus_interact_output/`.

Results of interacting between two agents will be saved in `user_interact_output/`.


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

## Transfer Learning
### Create transfer learning data
```console
>>> bash src/prepare_data_transfer_learning.sh
```
Data for single-to-multi domain setup will be created in `data/single_to_multi/`.
Data for domain adaptation setup will be created in `data/domain_transfer/`.

Once the data is generated, one can run the model training in the following procedure:
1. train from scratch, with the running mode `pretrain`.
2. fine-tune the model using the adaption data, with the running mode `finetune`.
3. fine-turn the model using reinforcement learning, with the running mdoe `rl`.

(Bash scripts of running transfer learning will be added later)

## Citation
```bibtex
@inproceedings{tseng-etal-2021-transferable,
    title = "Transferable Dialogue Systems and User Simulators",
    author = "Tseng, Bo-Hsiang  and
      Dai, Yinpei  and
      Kreyssig, Florian  and
      Byrne, Bill",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.13",
    pages = "152--166",
}
```

## Contact Us
Please contact bht26@cam.ac.uk or raise an issue in this repository.
