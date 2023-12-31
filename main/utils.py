import cv2 as cv 
import numpy as np
import math


# values =(blue, green, red) opencv accepts BGR values 
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
CYAN = (255,255,0)
YELLOW =(0,255,255)
MAGENTA = (255,0,255)
GRAY = (128,128,128)
GREEN = (0,255,0)
PURPLE = (128,0,128)
ORANGE = (0,165,255)
PINK = (147,20,255)
points_list =[(200, 300), (150, 150), (400, 200)]


    
def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):
    
    (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    cv.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    cv.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text

    return img

def textWithBackground(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3, bgOpacity=0.5):
    
    (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    overlay = img.copy() # coping the image
    cv.rectangle(overlay, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    new_img = cv.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0) # overlaying the rectangle on the image.
    cv.putText(new_img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text
    img = new_img

    return img




# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord


def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance


def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
 
    
    upper_lid_top = landmarks[right_indices[2]][1]  # Upper eyelid top
    upper_lid_bottom = landmarks[right_indices[5]][1]  # Upper eyelid bottom
    lower_lid_top = landmarks[right_indices[10]][1]  # Lower eyelid top
    lower_lid_bottom = landmarks[right_indices[13]][1] 
    
    if upper_lid_top == upper_lid_bottom:
        blink_amplitude = 0  # Set amplitude to zero when denominator is zero
    else:
        blink_amplitude = (upper_lid_top - lower_lid_bottom) / (upper_lid_top - upper_lid_bottom)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)
    
    
    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance
    
    ratio = (reRatio+leRatio)/2

    return ratio, blink_amplitude



# Eyes function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

 
    eyes = cv.bitwise_and(gray, gray, mask=mask)

    eyes[mask==0]=155
    
  
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left




# Constants for eye position estimation
POSITION_THRESHOLD = 0.4  # Threshold for considering a part as the dominant eye position

# Eye positions
EYE_POSITIONS = {
    "RIGHT": {
        "color": (0, 255, 0),  # Green
        "parts": [0],
        "direction": "Right",
    },
    "CENTER": {
        "color": (0, 255, 255),  # Yellow
        "parts": [1],
        "direction": "Center",
    },
    "LEFT": {
        "color": (255 , 0 , 0),  # Gray
        "parts": [2],
        "direction": "Left",
    },
    "CLOSED": {
        "color": (128, 128, 128),  # Gray
        "parts": [],
        "direction": "Closed",
    },
}

#Eyes position estimator
def positionEstimator(cropped_eye):
    # Getting height and width of the eye
    h, w = cropped_eye.shape

    # Remove noise from the image
    gaussian_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv.medianBlur(gaussian_blur, 3)

    # Apply thresholding to convert to binary image
    _, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # Divide the eye into three parts with adjusted division points
    piece1 = int(w * 0.3)
    piece2 = int(w * 0.6)

    # Slice the eye into three parts
    left_part = threshed_eye[:, :piece1]
    center_part = threshed_eye[:, piece1:piece2]
    right_part = threshed_eye[:, piece2:]

    # Count the black pixels in each part
    left_pixels = np.sum(left_part == 0)
    center_pixels = np.sum(center_part == 0)
    right_pixels = np.sum(right_part == 0)

    # Calculate the ratios of black pixels in each part
    total_left_pixels = left_part.shape[0] * left_part.shape[1]
    total_center_pixels = center_part.shape[0] * center_part.shape[1]
    total_right_pixels = right_part.shape[0] * right_part.shape[1]

    left_ratio = left_pixels / total_left_pixels
    center_ratio = center_pixels / total_center_pixels
    right_ratio = right_pixels / total_right_pixels

    # Find the dominant eye position
    max_ratio = max(left_ratio, center_ratio, right_ratio)
    eye_position = "CLOSED"

    if max_ratio >= POSITION_THRESHOLD:
        if max_ratio == left_ratio:
            eye_position = "LEFT"
        elif max_ratio == center_ratio:
            eye_position = "CENTER"
        elif max_ratio == right_ratio:
            eye_position = "RIGHT"

    return eye_position, EYE_POSITIONS[eye_position]["color"], EYE_POSITIONS[eye_position]["direction"]

