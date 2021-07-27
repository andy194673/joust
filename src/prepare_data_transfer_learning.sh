#!/bin/bash

# prepare single-domain to multi-domain data
#mkdir -p data/single_to_multi/
#python utils/split_single_multi_data.py
#echo 'Done creating single-to-multi domain data, stored in data/single_to_multi/'
# TODO: back

# prepare domain transfer data
mkdir -p data/domain_transfer
python utils/split_domain_transfer_data.py
echo 'Done creating domain transfer data, stored in data/domain_transfer'