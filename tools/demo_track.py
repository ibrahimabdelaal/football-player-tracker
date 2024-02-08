import argparse
import os
import os.path as osp
import time
import sys

import cv2
import torch
import numpy as np
from loguru import logger
import sys
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import h5py
from torchvision.models import resnet152
import torch
import torch.nn as nn
import torchvision.models as models
sys.path.remove('g:\\year5_spring\\track\\bytetrack\\bytetrack_reid')
sys.path.append(r'G:/Year5_Spring/track/Tracking/station/')
sys.path.append('G:\Year5_Spring\\track\\Tracking\\station\\tools\\demo_track.py\\src\\cython-bbox')


from yolox.data.data_augment import preproc,preproc2
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker1
from yolox.tracker.byte_tracker_FairMOT import BYTETracker
from yolox.tracking_utils.timer import Timer
from merge_utlis import Eye_bird,draw_2d,extract_features,ball_feature,draw_3d,draw_3d2
#sys.path.append(r'/home/dell/.local/lib/python3.8/site-packages')
from elements.perspective_transform import Perspective_Transform

from models.common import DetectMultiBackend
# sys.path.remove(r'/home/dell/.conda/envs/res_id/lib/python3.8/site-packages')

import torch.nn.functional as F
#sys.path.append(r'/home/dell/.conda/envs/res_id/lib/python3.8/site-packages')
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

##python tools/demo_track.py video --path G:\Year5_Spring\track\Bytetrack\ByteTrack\datasets\train11\tf.mp4 -f exps/yolox_s.py -c pretrained/best_ckpt.pth   --fuse --save_result -cc pretrained/best.pt --device --fp
16
# python3 tools/demo_track.py image --path datasets/mot/test/MOT17-01-DPM -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result
#python tools/demo_track.py video --path G:\Year5_Spring\track\Bytetrack\ByteTrack\datasets\train12\test2.mp4 -f exps/yolox_s.py -c pretrained/last.pt --fp16 --fuse --save_result
def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--bird",
        action="store_true",
        help="whether to draw Bird eyeview on the image or not ",
    )
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-cc", "--ckptt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-out", "--output", default=None, type=str, help="save path for output")
    parser.add_argument("-resnet_h5_weight", "--res", default=None, type=str, help="resnet weight")

    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=.35, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=60, type=int, help="frame rate (fps)")
    parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
    parser.add_argument("--imsize", type=int, default=640, help="Yolo input image size")

    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.1, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.75, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=20,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=0, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info


class Predictor2(object):
    def __init__(
        self,
        model,
        num_classes,
        confthre,
        nms_thresh, 
        test_size,
        fp16,
        device,
        trt_file=None,
        decoder=None,
        
        
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = num_classes
        self.confthre =confthre
        self.nmsthre = .4
        self.test_size = (test_size,test_size)
        self.device =device
        self.fp16 = fp16
        ###print("test_size ",test_size)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, self.test_size[0], self.test_size[1]), device=self.device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        #cv2.imwrite("predict.jpg",img)
       # ###print(ratio)
        img, ratio = preproc2(img, self.test_size, self.rgb_means, self.std)
        ####print("after processing ",img.shape)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.fp16 else img.float() 

        if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
        with torch.no_grad():
            timer.tic()
            outputs = self.model(img,augment=False)
            outputs = postprocess(outputs[0], self.num_classes,self.confthre,self.nmsthre)
           
           
        return outputs, img_info



def image_demo(predictor2,resmodel ,vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs2, img_info2 = predictor2.inference(img_path, timer)
        crop_and_save_boxes(outputs2[0],img_info2['raw_img'],'cut',[img_info2['height'], img_info2['width']],exp.test_size)
        img_info=img_info2
        oo=outputs2[0].cpu().numpy()
          
        o=extract_features(resmodel,outputs2[0], img_path,[img_info['height'], img_info['width']],exp.test_size)
        id_feature =o[:, 7:]  # [detect_num, 128]
        id_feature = F.normalize(id_feature, dim=1)  # normalization of id embeddings
        id_feature = id_feature.cpu().numpy()
        
        if outputs2[0] is not None:
            online_targets = tracker.update(o, [img_info['height'], img_info['width']], exp.test_size, id_feature)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    ###print("after update ",tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")






def imageflow_demo(predictor2,resmodel ,vis_folder, current_time, args,bf,color_model):
    print("start tracking.......")
    start = time.time()
    perspective_transform = Perspective_Transform()
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
         
        save_path = osp.join(save_folder, args.path.split("\\")[-1])

    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"),  fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    tracker2=BYTETracker1(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []

    ## Load the field persepective image ####

    bg_ratio = int(np.ceil(width*1/(3*115)))
    ###print(bg_ratio)
    gt_img = cv2.imread('inference/black.jpg')
    green_img= cv2.imread('inference/green.jpg')
   # ###print("image is  Noene",gt_img==None)
    cv2.imwrite('test_im.jpg',gt_img)
    gt_img = cv2.resize(gt_img,(115*bg_ratio,74*bg_ratio))
    green_img2=cv2.resize(green_img,(gt_img.shape[1],gt_img.shape[0]))
    gt_h, gt_w, _ = gt_img.shape

    ######
    assigned_col={}      # to store the colors assigned for players during the match
    img_list=[]
    id_list=[]
    coord_list=[]
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            bg_img = gt_img.copy()
            g_img=green_img2.copy()
           
            outputs2, img_info2 = predictor2.inference(frame, timer)  #yolo output
            img_info=img_info2
            ##print("outpits ",outputs2[0][:, 6])
            rows_to_delete = outputs2[0][:, 6] == 0.0000e+00      # select ball==class 0
            selected_rows = outputs2[0][outputs2[0][:, 6] == 0.0000e+00]    

            # Convert the selected rows to a tensor
            fairmot = torch.tensor(selected_rows)
          
           
            # Delete rows from the original tensor
            bytetrack =outputs2[0][~rows_to_delete]
            #extract feature using resnet model
            objects_detected=extract_features(resmodel,fairmot,frame,[img_info['height'], img_info['width']],(args.imsize,args.imsize),bf,args.device)
            id_feature =objects_detected[:, 7:]  # [detect_num, 128]
            id_feature = F.normalize(id_feature, dim=1)  # normalization of id embeddings
            id_feature = id_feature.cpu().numpy()
            frame1=frame.copy()
            if outputs2[0] is not None:

                online_targets = tracker.update(objects_detected, [img_info['height'], img_info['width']], (args.imsize,args.imsize), id_feature) # update the ball using features to avoid false positive
                online_targets2=tracker2.update(bytetrack, [img_info['height'], img_info['width']], (args.imsize,args.imsize))  # update players 
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_classes=[]
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = 1000
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_classes.append(t.classes)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                for t in online_targets2:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_classes.append(t.classes)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
                )
                
            '''
            start drawing the eye bird view after getting the tracklets
            '''

            if args.bird :
                if frame_id % 10 ==0: # Calculate the homography matrix every 5 frames
                            M, warped_image = perspective_transform.homography_matrix(frame1)
                

                #prepare the tracklets for 2d img
                
                outs=Eye_bird(online_tlwhs,online_ids,online_scores,online_classes,img_info,frame1)
                width=frame1.shape[1]
                height=frame1.shape[0]

                '''
                draw the 2d eye bird img
                '''
                g_img=draw_3d2(outs,width,height,frame1,g_img,M,gt_h, gt_w,bg_ratio,img_info,assigned_col,color_model, img_list,
                    id_list,coord_list)

            if args.save_result and args.bird:
                online_im[online_im.shape[0]-bg_img.shape[0]:, online_im.shape[1]-bg_img.shape[1]:] = g_img
                vid_writer.write(online_im)
            else :
                vid_writer.write(online_im)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
        
            break
        frame_id += 1

    if args.save_result:
        print ("it took", time.time() - start, "seconds.")
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
        import torch
    import torch
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
    
        logger.info("loaded checkpoint done.")
        model2 = DetectMultiBackend(args.ckptt, device=args.device)
        model2.eval()

    if args.fuse:
        logger.info("\tFusing model...")
        #model = fuse_model(model)
        model2 = fuse_model(model2)

    if args.fp16:
        #model = model.half()  # to FP16
        model2 = model2.half()  # to FP16


    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model2.head.decode_in_inference = False
        decoder = model2.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor2 = Predictor2(model2, args.num_classes,args.conf,args.nms,args.imsize,args.fp16,args.device,trt_file, decoder)
   

    class ResNet50(nn.Module):
        def __init__(self):
            super(ResNet50, self).__init__()
            self.model = models.resnet152(pretrained=False)
            self.model.fc = nn.Identity()
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, 128)

        def forward(self, x):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            x = x.view(1, 128)  # Reshape to (128, 1)

            return x

    # Instantiate the ResNet50 model for color extraction
    resmodel = ResNet50()

# Load the weights from the H5 file
    h5_weights = h5py.File(args.res, 'r')

  
    state_dict2 = resmodel.state_dict()
    for k, v in state_dict2.items():
        if k in h5_weights:
            param = torch.from_numpy(h5_weights[k][...])
            v.copy_(param)
    resmodel.eval()
    resmodel.to(args.device)
    ball_f=ball_feature(resmodel,args.device)
    current_time = time.localtime()
    
    if args.demo == "image":
        image_demo(predictor2 ,resmodel,vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor2,resmodel ,vis_folder, current_time, args,ball_f,resmodel)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)

