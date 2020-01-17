#!/bin/bash

# This is necessary for conda activate to work properly
. /proj/anaconda3/etc/profile.d/conda.sh

# Activate smallfry env
printf "conda activate ml\n"
conda activate ml

printf "\nconda env list\n"
conda env list

printf "which python\n"
which python

# Execute the command that was passed in as a string to this script.
printf "\nExecute command: '$1'\n"
eval $1