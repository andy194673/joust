#!/bin/bash

# create ckp, log folder
for type in 'checkpoint' 'log'; do
  for mode in 'pretrain' 'finetune' 'rl'; do
		mkdir -p $type/$mode
	done
done

# create result folder
for type in 'result' 'result_usr'; do
	for mode in 'pretrain' 'finetune' 'rl'; do
		mkdir -p $type/$mode/word
		mkdir -p $type/$mode/act
		mkdir -p $type/$mode/dst
	done
done
