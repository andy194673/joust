#!/bin/bash
# prepare single-domain to multi-domain data
mkdir -p data/single_to_multi/
python utils/split_single_multi_data.py
echo 'Done creating single-to-multi domain data, stored in data/single_to_multi/'
exit


# prepare domain transfer data
mkdir -p data/domain_transfer
#python3 utils/split_domain_transfer_data.py > info/split_domain_transfer.log
#echo 'Done'