#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate res_id

cd "/home/dell/ByteTrack_HOME/station/"

python tools/demo_track.py video --path test4.mp4 -f /home/dell/ByteTrack_HOME/station/exps/yolox_s.py -c pretrained/best_ckpt.pth   --fuse --save_result -cc pretrained/Med.pt --output /media/dell/HDD/GradPro/output/