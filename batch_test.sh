#!/bin/bash
for id in $( seq 101 120 );do
checkpoint="/public/home/jm/Data/output/stm_output/models_with_coco/DAVIS17/recurrent_${id}.pth.tar"
echo "TEST(${id}):" $checkpoint
python test.py --checkpoint ${checkpoint}
done