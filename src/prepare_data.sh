#!/bin/bash
cd data/
mkdir -p process_data/
tar zxvf raw_data.tar.gz
tar zxvf dst.tar.gz
cd ..
python data_preprocess/create_delex_data.py
echo "Done Data Pre-processing!"

