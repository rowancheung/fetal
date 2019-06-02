#!/bin/bash

export HOME=/data/vision/polina/projects/fetal_brain
fetal_dir=$HOME

cd ${fetal_dir}/fetal
python_exe=${fetal_dir}/ENV/bin/python

${python_exe} time_course.py 
