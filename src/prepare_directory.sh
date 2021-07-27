#!/bin/bash

# create result file
for folder_type in 'result' 'result_usr'; do
  for result_type in 'dst' 'policy' 'nlg';
		mkdir -p $folder_type/$result_type
	done
done
