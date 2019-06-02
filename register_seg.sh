#! /bin/bash

subject=$1
numFrames=$2

segDir='/data/vision/polina/projects/placenta_segmentation/data/predict_cleaned/unet3000/'$subject'/'
regDir='/data/vision/polina/projects/fetal_brain/fetal/registered/'$subject'/'
outputDir='/data/vision/polina/projects/fetal_brain/fetal/registered_segmentations/'$subject'/'
export PATH=$PATH:/data/vision/polina/shared_software/ANTS/build/bin

if [ ! -d $outputDir ]; then
  mkdir -p $outputDir
fi

for i in  $(seq -w 0001 $numFrames)
do
string='WarpImageMultiTransform 3 '$segDir$subject'_'$i'.nii.gz '$outputDir$subject'_'$i'.nii.gz -R '$segDir$subject'_'$i'.nii.gz --use-NN'

img=$subject'_0143.nii.gz_to_'$subject'_'${i}'.nii.gz_inverseWarp.nii.gz'

string=$string' '$regDir$img

$string
done


