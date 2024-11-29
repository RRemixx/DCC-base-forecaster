#!/bin/sh

# remove tmp yaml files
rm ../setup/cp_exp_params/*_*.yaml

rm ../setup/cp_exp_params/1001.yaml
rm ../setup/cp_exp_params/1002.yaml
rm ../setup/cp_exp_params/1003.yaml
rm ../setup/cp_exp_params/1004.yaml
rm ../setup/cp_exp_params/1005.yaml

rm *.out

rm ../src/forecaster/*.out

# remove results files
# rm ../results/*.pkl

# remove model files
rm ../models/*