#!/bin/bash

placenta_dir=/data/vision/polina/projects/placenta_segmentation

###################

scripts/download_data.sh nifti/$1/
python -B split_nifti.py $1
scripts/upload_data.sh raw/$1/
