#!/bin/bash

# note: to run this bash file with 7 cores in parallel: sh run.sh 7
# python3 src/config.py
mkdir -p data result/summary_pca result/summary_xx final
file_list=file_list_random.txt
nworkers=$1
cat $file_list | xargs -L 1 -P $nworkers -I {} python3 src/main.py {}
python3 -W ignore src/summarize.py #ignore RuntimeWarning
