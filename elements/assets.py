import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from tensorflow.keras.applications.resnet import  preprocess_input


pallete = {
    'b': (0, 0, 128),
    'r': (255, 0, 0),
    'c': (0, 192, 192),
    'm': (192, 0, 192),
    'y': (192, 192, 0),
    'k': (0, 0, 0),
     'w': (255, 255, 255)
}

# pallete = {
#     'b': (0, 0, 128),
#     'r': (255, 0, 0),
#     'y': (192, 192, 0)
   
# }
color_ranges = [
    ((0, 0, 0), (85, 85, 85)),
    ((85, 85, 85), (170, 170, 170)),
    ((170, 170, 170), (255, 255, 255)),
    # Add an additional range
    ((255, 255, 255), (255, 255, 255))
]

custom_colors = [
    (255, 0, 0),    # Custom color 1: Red
    (0, 0, 255),    # Custom color 2: Blue
    (255, 255, 0),  # Custom color 3: Yellow
    (128, 128, 128)  # Custom color 4: New color for the additional range
]

color_for_labels = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in color_for_labels]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def transform_matrix(matrix, p, vid_shape, gt_shape):
    p = (p[0]*1280/vid_shape[1], p[1]*720/vid_shape[0])
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))

    p_after = (int(px*gt_shape[1]/115) , int(py*gt_shape[0]/74))

    return p_after
def edges(img):
    import cv2
    import numpy as np

    # Read the image
    image = img

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian smoothing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection with threshold values
    edges = cv2.Canny(blurred, 40,60)

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find the outermost contour representing the edges
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the region inside the edges
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [outer_contour], 0, (255), thickness=cv2.FILLED)

    # Set the region outside the edges to black
    result = cv2.bitwise_and(image, image, mask=mask)
    return result 

# Color Detection with K-means
def detect_color(img,id,assigned_col,model):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img=preprocess_input(img)
    # #print("Res model prediction.................")
    # color_features = model.predict(np.expand_dims(img, axis=0))
    # #print("Res model end.................")
    # num_pixels = color_features.shape[1] * color_features.shape[2]
    # #print("reshaping.....................")
    # img = color_features.reshape((num_pixels, color_features.shape[3]))
    # ##print(img)
    # #img = img.reshape((img.shape[1]*img.shape[0],3))
    # #print("Kmeans .......................")
    img = img.reshape((img.shape[1]*img.shape[0],3))
    # hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # hist = cv2.normalize(hist, hist).flatten()
    # #print(hist.shape)
    kmeans = KMeans(n_clusters=3,n_init=10)
    s = kmeans.fit(img)

    labels = kmeans.labels_
    centroid = kmeans.cluster_centers_
    labels = list(labels)
    percent=[]
    
    for i in range(len(centroid)):
        j=labels.count(i)
        j=j/(len(labels))
        percent.append(j)
    detected_color = centroid[np.argmin(percent)]
    
    list_of_colors = list(pallete.values())
    assigned_color = closest_color(list_of_colors, detected_color)[0]
    #assigned_color = assign_color_to_range(color_ranges, custom_colors,(int(assigned_color[2]), int(assigned_color[1]), int(assigned_color[0]))  ) 
    #print(assigned_color)
    assigned_color = (int(assigned_color[2]), int(assigned_color[1]), int(assigned_color[0]))
    # if assigned_color == (0, 0, 0):
    #     assigned_color = (128, 128, 128)
    
    
    # if assigned_color == (0, 0, 0):
    #     assigned_color = (128, 128, 128)


    #Assign color based on minimum distance within threshold
    team1_color = (0, 255, 255)  # Red
    team2_color = (255, 0, 0)  # Green
    referee_color = (0, 255, 255)  # Blue
    threshold = 200
    dist_team1 = np.linalg.norm(np.array(detected_color) - np.array(team1_color))
    dist_team2 = np.linalg.norm(np.array(detected_color) - np.array(team2_color))
    #dist_referee = np.linalg.norm(np.array(detected_color) - np.array(referee_color))
    if dist_team1 < dist_team2 :
        assigned_color = team1_color
    # else :
    #     assigned_color = team2_color
    else:
        assigned_color = team2_color
    assigned_col[id]=assigned_color
    return assigned_color
    
def detect_color2(img,id,assigned_col,model):
    # color_ranges = [((0, 0, 0), (85, 85, 85)), ((85, 85, 85), (170, 170, 170)), ((170, 170, 170),(255, 255, 255))]

    # assigned_values = [
    # (0, 0, 0),                       # Black (RGB values)
    # (255, 0, 0),                     # Red (RGB values)                     # Green (RGB values)
    # (0, 0, 255)                      # Blue (RGB values)
    #        ]

    #img=edges(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[1]*img.shape[0],3))

    kmeans = KMeans(n_clusters=2)
    s = kmeans.fit(img)

    labels = kmeans.labels_
    centroid = kmeans.cluster_centers_
    labels = list(labels)
    percent=[]
    
    for i in range(len(centroid)):
        j=labels.count(i)
        j=j/(len(labels))
        percent.append(j)
    detected_color = centroid[np.argmin(percent)]
    
    list_of_colors = list(pallete.values())
    assigned_color = closest_color(list_of_colors, detected_color)[0]
    #assigned_color = assign_color_to_range(color_ranges, custom_colors,(int(assigned_color[2]), int(assigned_color[1]), int(assigned_color[0]))  ) 
    assigned_color = (int(assigned_color[2]), int(assigned_color[1]), int(assigned_color[0]))
    if assigned_color == (0, 0, 0):
        assigned_color = (128, 128, 128)
    
    
    if assigned_color == (0, 0, 0):
        assigned_color = (128, 128, 128)


    # Assign color based on minimum distance within threshold
    team1_color = (255, 0, 0)  # Red
    team2_color = (0, 255, 0)  # Green
    referee_color = (0, 0, 255)  # Blue
    threshold = 300
    dist_team1 = np.linalg.norm(np.array(detected_color) - np.array(team1_color))
    dist_team2 = np.linalg.norm(np.array(detected_color) - np.array(team2_color))
    dist_referee = np.linalg.norm(np.array(detected_color) - np.array(referee_color))
    #print("distances ",dist_team1,dist_team2,dist_referee)
    if dist_team1<dist_team2+10 and dist_team1<dist_referee+10 and dist_team1<threshold:
        assigned_color = (150, 50, 100)
    elif dist_team2<dist_team1 and dist_team2<dist_referee-20 and dist_team2<threshold:
         assigned_color = (20, 150, 155)

    elif dist_referee<dist_team1 and dist_referee<dist_team2  and dist_referee<threshold:
        assigned_color = (128, 128, 128)


    # if dist_team1 <= threshold:
    #     assigned_color = team1_color
    # elif dist_team2 <= threshold:
    #     assigned_color = team2_color
    # elif dist_referee <= threshold:
    #     assigned_color = referee_color
    # else:
    #     #print("entring else ")
    #     assigned_color = detected_color
    ##print("here is assigned color :",assigned_color)
    assigned_col[id]=assigned_color
    return assigned_color



# Find the closest color to the detected one based on the predefined palette
def closest_color(list_of_colors, color):
    colors = np.array(list_of_colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_shortest = np.where(distances==np.amin(distances))
    shortest_distance = colors[index_of_shortest]
    return shortest_distance 


import numpy as np

# def closest_color(list_of_colors, color):
#     colors = np.array(list_of_colors)
#     color = np.array(color)
#     distances = np.sqrt(np.sum((colors - color)**2, axis=1))
#     index_of_shortest = np.argmin(distances)
#     shortest_distance = colors[index_of_shortest]
#     return shortest_distance



import numpy as np

import numpy as np


def assign_color_to_range(color_ranges,  custom_colors,assigned_color):
    assigned_color = np.array(assigned_color)  # Convert assigned_color to numpy array for calculations

    closest_range_idx = 0
    closest_range_dist = float('inf')

    # Find the closest range
    for i, (range_min, range_max) in enumerate(color_ranges):
        range_min = np.array(range_min)  # Convert range_min to numpy array for calculations
        range_max = np.array(range_max)  # Convert range_max to numpy array for calculations

        dist = np.linalg.norm(assigned_color - range_min) + np.linalg.norm(assigned_color - range_max)

        if dist < closest_range_dist:
            closest_range_dist = dist
            closest_range_idx = i

    # Assign the assigned_color to the corresponding custom color based on the closest range
    assigned_custom_color = custom_colors[closest_range_idx]

    return assigned_custom_color


    
    
    









