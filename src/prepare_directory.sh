#!/bin/bash

# create ckp, log folder
#for folder_type in 'checkpoint' 'log'; do
#		mkdir $folder_type
#done

# create result file
for folder_type in 'result' 'result_usr'; do
  for result_type in 'dst' 'policy' 'nlg';
		mkdir -p $folder_type/$result_type
	done
done
