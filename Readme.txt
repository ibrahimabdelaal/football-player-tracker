pip3 install -r requirements.txt
python3 setup.py develop
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
pip install pyflann




## Run using this command 
cd main_folder
python tools/demo_track.py video --path test4.mp4 -f /home/dell/ByteTrack_HOME/station/exps/yolox_s.py -c pretrained/best_ckpt.pth   --fuse --save_result -cc pretrained/Med.pt --output /media/dell/HDD/GradPro/output/
change --path to your video 