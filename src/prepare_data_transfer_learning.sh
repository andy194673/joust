#!/bin/bash

# prepare single-domain to multi-domain data
rm -r data/single_to_multi/
mkdir -p data/single_to_multi/
python utils/split_single_multi_data.py
echo 'Done creating single-to-multi domain data, stored in data/single_to_multi/'

# prepare domain transfer data
rm -r data/domain_transfer
mkdir -p data/domain_transfer
python utils/split_domain_transfer_data.py
echo 'Done creating domain transfer data, stored in data/domain_transfer'