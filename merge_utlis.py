import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
from loguru import logger
import sys
import cv2
from elements.assets import transform_matrix, detect_color,detect_color2
from utils.plots import plot_one_box
import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.resnet import  preprocess_input
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.decomposition import PCA
import cv2
import torch
import torchvision.transforms as transforms


def ball_feature(model,device):
    image = cv2.imread('s.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
     
    input_tensor = transform(image)
# Add a batch dimension to the input tensor
    input_tensor = input_tensor.unsqueeze(0)
    
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        
        features = model(input_tensor)
    features = features.cpu().numpy()
    ##print("output features")
    return features



def detect_utlis(im_list, model):
    features_list = []
    
    for image in im_list:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preprocessed_image = preprocess_input(image)
            features = model.predict(np.expand_dims(preprocessed_image, axis=0))
            features = features.reshape((128, 1))
            ##print(features.shape)
            ###print(features)
            
            features_list.append(features.flatten())
    ###print(features_list)
        
    
    features_array = np.array(features_list)
    features_array = features_array.reshape(features_array.shape[0], -1)
    ##print(features_array.shape,len(features_list),features_array[0])
    
    num_clusters = 3
    color_labels = [(255, 0, 0), (0, 0, 255), (255, 255, 0)]
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(features_array)

    # Get the cluster centers (representing dominant colors)
    cluster_centers = kmeans.cluster_centers_

    # Map cluster labels to color labels
    centroid_color_map = {label: color_labels[label] for label in kmeans.labels_}
    
    # Assign color labels to each image
    image_colors = [centroid_color_map[label] for label in kmeans.labels_]

    return image_colors


def assign_to_cluster(new_image, features_array, kmeans, centroid_color_map):
    # Compute color histogram for the new image
    hist = cv2.calcHist([new_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Exclude the grass area from the color histogram
    roi_x1, roi_y1, roi_x2, roi_y2 = 100, 100, 900, 600
    grass_area = new_image[roi_y1:roi_y2, roi_x1:roi_x2]
    grass_hist = cv2.calcHist([grass_area], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    grass_hist = cv2.normalize(grass_hist, grass_hist).flatten()
    hist -= grass_hist

    # Calculate distances to each old cluster
    distances = np.linalg.norm(features_array - hist, axis=1)

    # Find the index of the closest cluster
    closest_cluster_idx = np.argmin(distances)

    # Assign the new image to the closest cluster
    new_image_color = centroid_color_map[closest_cluster_idx]

    return new_image_color


import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def detect_utils1(im_list,model):
    features_list = []
    
    for image in im_list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Compute color histogram for the entire image
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Exclude the green grass from the color histogram
        green_range = np.array([36, 0, 0]), np.array([70, 255, 255])
        mask = cv2.inRange(image, green_range[0], green_range[1])
        indices = np.where(mask > 0)[0]  # Extract the first element from the tuple
        hist[indices] = 0
        
        features_list.append(hist)
    
    features_array = np.array(features_list)
    
    num_clusters = 3
    color_labels = [(255, 0, 0), (0, 0, 255), (255, 255, 0)]
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(features_array)

    # Get the cluster centers (representing dominant colors)
    cluster_centers = kmeans.cluster_centers_

    # Map cluster labels to color labels
    centroid_color_map = {label: color_labels[label] for label in kmeans.labels_}
    
    # Assign color labels to each image
    image_colors = [centroid_color_map[label] for label in kmeans.labels_]
    
    return image_colors

  


def Eye_bird (online_tlwhs,online_ids,online_scores,online_classes,img_info,frame):
        outs=[]
        width=img_info['width']
        height=img_info['height']
        classes={0 : "ball" ,1:"player" ,2:"player",3:"player"}
        for i,j,k,l in zip(online_tlwhs,online_ids,online_scores,online_classes):
            if int(l) in list(classes.keys()):
               
                score = np.round(k)
                label = classes[int(l)]
                img_h, img_w = img_info['height'], img_info['width']
                bboxes=tlwh_to_tlbr(i)
                xmin=bboxes[0]
                ymin=bboxes[1]
                xmax=bboxes[2]
                ymax=bboxes[3]
                #cv2.imshow("color",frame[int(ymin):int(ymax), int(xmin):int(xmax)])
                item = {'label': label,
                        'bbox' : [(int(xmin),int(ymin)),(int(xmax),int(ymax))],
                        'score': score,
                        'cls' : int(l),
                        'id' :int(j)}
                ###print(item)
                outs.append(item)
        return outs


def crop_and_save_boxes(boxes, image, save_path,info,img_size):
    bboxess=boxes
    bboxess=bboxess.cpu().numpy()
    boxes=bboxess[:, :4].copy()
    img_h, img_w = info[0], info[1]  # origin image size

    scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))  # input/origin, <1
    boxes /= scale  # map bbox to origin image size
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        ##print("features",box)
        # Ensure box coordinates are within image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.shape[1], int(x2))
        y2 = min(image.shape[0], int(y2))

        # Crop the box from the image
        cropped_image = image[y1:y2, x1:x2]

        # Save the cropped image
        filename = f"box_{i}.jpg"
        save_pathh = f"{save_path}/{filename}"
        cv2.imwrite(save_pathh, cropped_image)

       # ##print(f"Box {i}: Cropped image saved as {save_path}")

def extract_features(model, boxess,image1,info ,img_size,ball_f,device):

    image=image1.copy()
    bboxess=boxess
    bboxess=bboxess.cpu().numpy()
    scores = bboxess[:, 4]
    cls_conf=bboxess[:, 5]
    cls=bboxess[:, 6]
    boxes=bboxess[:, :4].copy()
    orig=bboxess[:, :4]
    out=np.zeros((bboxess.shape[0],128+6))
    img_h, img_w = info[0], info[1]  # origin image size
    scale = min(img_size[1] / float(img_h), img_size[0] / float(img_w))  # input/origin, <1
    boxes /= scale  # map bbox to origin image size
    l=[]
    all_f=[]
    
    for i,box in enumerate(boxes):
        temp=[]
        # Extract box coordinates
        x1, y1, x2, y2 = box
        
        # Crop the box from the image
        #print("here is frame size ",image.shape)
        #print("bounding box ",box)
        if cls[i]==0:
            off=0
        else:
            off=0
        cropped_image = image[int(y1-off):int(y2+off), int(x1-off):int(x2+off)]
        import time
        #Preprocess the cropped image
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224, 224))
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
        input_tensor = transform(cropped_image)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)
       
       
        
        # Obtain the features
        with torch.no_grad():
            features = model(input_tensor)
        features = features.cpu().numpy()
        all_f.append(features)
        temp.extend(orig[i].tolist())
        ###print(temp)
        temp.append(scores[i])
        temp.append(cls_conf[i])
        temp.append(cls[i])
        temp.extend(features[0].tolist())
        l.append(temp)
    ## calculate the distance
    from scipy.spatial.distance import euclidean
  
  
    if len(all_f)>1:
     third_feature_array = ball_f[0]
     import time
     distances=[]
     for i,f in  enumerate(all_f):
        
        feature_array2 = f[0]
        ##print("here in all f ",third_feature_array)
        
        distance = euclidean(feature_array2, third_feature_array)
        distances.append(distance)
     closest_index = np.argmin(distances)
    #  #print(distances)
     
     output=np.array(l[closest_index]).reshape((1,135))
     output[: ,7:]=ball_f[0].tolist()
    else:
     output=np.array(l).reshape((bboxess.shape[0],135))

    return torch.tensor(output)


def draw_2d(outs,width,height,frame1,g_img,M,gt_h, gt_w,bg_ratio,img_info,assigend_col,model, img_list,id_list,coord_list):
                lenth=[]
                for i, obj in enumerate(outs):
                    
                    xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
                    x_center = (xyxy[0] + xyxy[2])/2 
                    y_center = xyxy[3]
                    coords = transform_matrix(M, (x_center, y_center), (height, width), (gt_h, gt_w))
                    if obj['label'] == 'player' : 
                        #try:
                        if int(obj['id']) not in list(assigend_col.keys()):
                                
                                color = detect_color2(frame1[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]],obj['id'],assigend_col,model)
                                ##print("detecting color happened ",assigend_col,color)
                        else :
                            color=assigend_col[obj['id']]
                        cv2.circle(g_img, coords, bg_ratio + 8, color, -1)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text=str(obj['id'])
                        text_size, _ = cv2.getTextSize(text, font, 1, 1)
                        text_x = 256 - int(text_size[0]/2)
                        text_y = 256 + int(text_size[1]/2)
                        font_scale=.3
                        x = coords[0] - int(text_size[0] * font_scale / 2)
                        y = coords[1] + int(text_size[1] * font_scale / 2)
                        text_pos = (x,y)
                        cv2.putText(g_img, text, text_pos, font, font_scale, (255, 255,255), 1)
                        # except Exception as e:
                                # ##print(e)
                                # pass
                    elif obj['label'] == 'ball':
                        coords = transform_matrix(M, (x_center, y_center), (height, width), (gt_h, gt_w))
                        cv2.circle(g_img, coords, bg_ratio + 8, (255, 255, 255), -1)
                        plot_one_box(xyxy, img_info['raw_img'], (128, 128, 128), label="ball")
                  
                return g_img
 # @jit(nopython=True)
def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

import numpy as np
from filterpy.kalman import KalmanFilter


def draw_3d(outs, width, height, frame1, g_img, M, gt_h, gt_w, bg_ratio, img_info, assigend_col, model, img_list, id_list, coord_list):
    for i, obj in enumerate(outs):
        xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = xyxy[3]
        coords = transform_matrix(M, (x_center, y_center), (height, width), (gt_h, gt_w))

        if obj['label'] == 'player':
            if int(obj['id']) not in list(assigend_col.keys()):
                color = detect_color2(frame1[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]], obj['id'], assigend_col, model)
            else:
                color = assigend_col[obj['id']]

            # Initialize Kalman filter
            kalman_filter = cv2.KalmanFilter(4, 2)
            kalman_filter.statePost = np.array([coords[0], coords[1], 0, 0], dtype=np.float32)  # Initial state [x, y, dx, dy]
            kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]], dtype=np.float32)  # State transition matrix
            kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0]], dtype=np.float32)  # Measurement matrix
            kalman_filter.errorCovPost = np.eye(4, dtype=np.float32) * 3000  # Covariance matrix
            kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1  # Process noise covariance
            kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.9 # Measurement noise covariance

            # Kalman prediction
            kalman_pred = kalman_filter.predict()

            # Update Kalman filter with measured coordinates
            kalman_meas = np.array([[coords[0]], [coords[1]]], dtype=np.float32)
            kalman_update = kalman_filter.correct(kalman_meas)

            # Get updated coordinates from Kalman filter
            updated_coords = (int(kalman_update[0]), int(kalman_update[1]))

            # Draw the point on the image using the updated coordinates
            cv2.circle(g_img, updated_coords, bg_ratio + 7, color, -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text=str(obj['id'])
            text_size, _ = cv2.getTextSize(text, font, 1, 1)
            text_x = 256 - int(text_size[0]/2)
            text_y = 256 + int(text_size[1]/2)
            font_scale=.3
            x = coords[0] - int(text_size[0] * font_scale / 2)
            y = coords[1] + int(text_size[1] * font_scale / 2)
            text_pos = (x,y)
            cv2.putText(g_img, text, text_pos, font, font_scale, (255, 255,255), 1)

            # Draw other annotations or perform additional operations on the image

        elif obj['label'] == 'ball':
            coords = transform_matrix(M, (x_center, y_center), (height, width), (gt_h, gt_w))
            cv2.circle(g_img, coords, bg_ratio + 7, (255, 255, 255), -1)
            plot_one_box(xyxy, img_info['raw_img'], (128, 128, 128), label="ball")
    return g_img






def draw_3d2(outs, width, height, frame1, g_img, M, gt_h, gt_w, bg_ratio, img_info, assigend_col, model, img_list, id_list, coord_list):
    field_width = gt_w# width of the field image
    field_height = gt_h# height of the field image
    boundary_threshold = 50

    for i, obj in enumerate(outs):
        xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = xyxy[3]
        coords = transform_matrix(M, (x_center, y_center), (height, width), (gt_h, gt_w))
        f_coords=[0,0]
        # Check if the coordinates are within the field boundaries
        
       
            
        if coords[1] >270:
                f_coords[1] = 270
                f_coords[0]=coords[0]
            # elif coords[1] > field_height - boundary_threshold:
            #         f_coords[1] = coords[1]-boundary_threshold
        elif coords[1]<0:
            
             f_coords[1] = 0
             f_coords[0]=coords[0]
        else:
            f_coords=coords
        if obj['label'] == 'player':
            if int(obj['id']) not in list(assigend_col.keys()):
                color = detect_color2(frame1[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]], obj['id'], assigend_col, model)
            else:
                color = assigend_col[obj['id']]

            # Initialize Kalman filter
            kalman_filter = cv2.KalmanFilter(4, 2)
            kalman_filter.statePost = np.array([f_coords[0], f_coords[1], 0, 0], dtype=np.float32)  # Initial state [x, y, dx, dy]
            kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]], dtype=np.float32)  # State transition matrix
            kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0]], dtype=np.float32)  # Measurement matrix
            kalman_filter.errorCovPost = np.eye(4, dtype=np.float32) * 3000  # Covariance matrix
            kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01  # Process noise covariance
            kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.9 # Measurement noise covariance

            # Kalman prediction
            kalman_pred = kalman_filter.predict()

            # Update Kalman filter with measured coordinates
            kalman_meas = np.array([[f_coords[0]], [f_coords[1]]], dtype=np.float32)
            kalman_update = kalman_filter.correct(kalman_meas)

            # Get updated coordinates from Kalman filter
            updated_coords = (int(kalman_update[0]), int(kalman_update[1]))

            # Draw the point on the image using the updated coordinates
            cv2.circle(g_img, updated_coords, bg_ratio + 8, color, -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text=str(obj['id'])
            text_size, _ = cv2.getTextSize(text, font, 1, 1)
            text_x = 256 - int(text_size[0]/2)
            text_y = 256 + int(text_size[1]/2)
            font_scale=.3
            x = f_coords[0] - int(text_size[0] * font_scale / 2)
            y = f_coords[1] + int(text_size[1] * font_scale / 2)
            text_pos = (x,y)
            cv2.putText(g_img, text, text_pos, font, font_scale, (255, 255,255), 1)

            # Draw other annotations or perform additional operations on the image

        elif obj['label'] == 'ball':
            #coords = transform_matrix(M, (x_center, y_center), (height, width), (gt_h, gt_w))
            cv2.circle(g_img, f_coords, bg_ratio + 8, (255, 255, 255), -1)
            plot_one_box(xyxy, img_info['raw_img'], (128, 128, 128), label="ball")
            font = cv2.FONT_HERSHEY_SIMPLEX
            text='ball'
            text_size, _ = cv2.getTextSize(text, font, 1, 1)
            text_x = 256 - int(text_size[0]/2)
            text_y = 256 + int(text_size[1]/2)
            font_scale=.2
            x = f_coords[0] - int(text_size[0] * font_scale / 2)
            y = f_coords[1] + int(text_size[1] * font_scale / 2)
            text_pos = (x,y)
            cv2.putText(g_img, text, text_pos, font, font_scale, (0, 0,0), 1)
   
    return g_img

            
def check_within_field_bounds(coords, field_width, field_height):
    x, y = coords

    # Define the field boundaries
    field_left = 0
    field_top = 0
    field_right = field_width
    field_bottom = field_height

    # Check if the coordinates are within the field boundaries
    if x >= field_left and x <= field_right and y >= field_top and y <= field_bottom:
        return True
    else:
        return False
