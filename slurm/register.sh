#!/bin/bash

export HOME=/data/vision/polina/projects/fetal_brain
fetal_dir=$HOME

cd ${fetal_dir}/fetal
python_exe=${fetal_dir}/venv/bin/python

${python_exe} registration.py 
