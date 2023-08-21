
import cv2 as cv
import mediapipe as mp
import time
import utils
import numpy as np

# variables 
initial_eye_closed = False
frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
prev_blink_time = 0


# constants
CLOSED_EYES_FRAME = 3
FONTS =cv.FONT_HERSHEY_COMPLEX
MICRO_SLEEP_DURATION_THRESHOLD = 0.5 # 
SLEEP_DURATION_THRESHOLD = 0.9  #
UNRESPONSIVE_DURATION_THRESHOLD = 1.0
closed_eyes_start_time = 0  #
opening_speed = 0

# Constants 
micro_sleep_start_time = 0
micro_sleep_detected = False
sleep_start_time = 0
sleep_detected = False


# Variables for unresponsive driver
unresponsive_start_time = 0
unresponsive_detected = False



# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh


last_face_data = None  # Variable to store the last known position and orientation of the face
is_tracking = True  # Variable to track if the head is currently being tracked
roi_x_min, roi_y_min, roi_x_max, roi_y_max = 200, 100, 1000, 700  # Example ROI coordinates
angle_threshold = 6
max_head_movement_per_frame = 6

###
# Function to check rapid head movement
def check_rapid_head_movement(curr_angles, prev_angles):
    # Check the difference in head angles
    yaw_diff = abs(curr_angles[0] - prev_angles[0])
    pitch_diff = abs(curr_angles[1] - prev_angles[1])

    # Check if the head movement exceeds the maximum allowed per frame
    if yaw_diff > max_head_movement_per_frame or pitch_diff > max_head_movement_per_frame:
        return True
    else:
        return False

def reinitialize_face_tracking():
    global last_face_data, is_tracking

    if last_face_data is not None:
        # Attempt re-initialization
        face_2d, face_3d, cam_matrix, dist_matrix, angles = last_face_data 

        # Update the camera matrix and distance matrix
        cam_matrix[0, 2] = img_h / 2  # Update the principal point x-coordinate
        cam_matrix[1, 2] = img_w / 2  # Update the principal point y-coordinate

        # Solve PnP
        success, rot_vec, trans_vec = cv.solvePnP(
            face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

        # Get the yaw and pitch angles in degrees
        yaw = angles[0] * 360
        pitch = angles[1] * 360

        # Check if the head yaw and pitch angles are within the desired range
        if abs(yaw) < angle_threshold and abs(pitch) < angle_threshold:
            is_tracking = True
        else:
            is_tracking = False

        # Only process the first detected face for re-initialization
        





####

# camera object 
camera = cv.VideoCapture(0)
camera_width = 1280
camera_height = 720
camera.set(cv.CAP_PROP_FRAME_WIDTH, camera_width)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, camera_height)

with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

        # Constants for blink duration calculation
    BLINK_THRESHOLD = 0.05  # Threshold to blink (in seconds)
    BLINK_HISTORY_SIZE = 3  #

    # Variables for blink duration calculation
    blink_start_time = 0
    blink_history = []
    blink_duration = 0 

    #blink frequency 
    blink_times = []  # List to store blink timestamps
    blink_frequency = 0

    # starting time 
    start_time = time.time()
    
    while True:
        frame_counter +=1 
        ret, frame = camera.read()  
        if not ret: 
            break 
        # Resizing frame
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            mesh_coords = utils.landmarksDetection(frame, results, False)

            # Check the number of visible eyes
            num_eyes = 0
            if len(mesh_coords) >= len(LEFT_EYE) or len(mesh_coords) >= len(RIGHT_EYE):
                num_eyes = 2
            elif len(mesh_coords) >= len(LEFT_EYE) or len(mesh_coords) >= len(RIGHT_EYE):
                num_eyes = 1

            results = face_mesh.process(frame)

            # To improve performance
            frame.flags.writeable = True

            # Convert the color space from RGB to BGR
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            img_h, img_w, img_c = frame.shape
            face_3d = []
            face_2d = []
            if is_tracking:
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        top_landmark_indices = [10, 152, 159, 145, 153, 144, 155, 151, 158]
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx in top_landmark_indices:
                                x, y = int(lm.x * img_w), int(lm.y * img_h)

                                # Get the 2D Coordinates
                                face_2d.append([x, y])

                                # Get the 3D Coordinates
                                face_3d.append([x, y, lm.z])
                    
                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w

                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                          [0, focal_length, img_w / 2],
                                          [0, 0, 1]])

                    # The Distance Matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    print("Number of 3D points:", len(face_3d))
                    print("Number of 2D points:", len(face_2d))

                    # Solve PnP
                    success, rot_vec, trans_vec = cv.solvePnP(
                        face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360

                    # Check if the head yaw and pitch angles are within the desired range
                    if -90 <= x <= 90 and -45 <= y <= 60:
                        # See where the user's head tilting
                        if y < -10:
                            text = "Looking Left"
                        elif y > 10:
                            text = "Looking Right"
                        elif x < -10:
                            text = "Looking Down"
                        elif 10 <= x <= 45:
                            text = "Looking Up"
                        else:
                            text = "Forward"
                    else:
                        text = "Head Out of Range"
                        is_tracking = False

                    # Check for rapid head movement
                    if last_face_data is not None:
                        prev_angles = last_face_data[4]  # Previous head angles
                        curr_angles = [angles[0], angles[1]]  # Current head angles

                        if check_rapid_head_movement(curr_angles, prev_angles):
                            # Rapid head movement detected, continue tracking without re-initialization
                            text += " (Rapid Movement)"
                        else:
                            # Normal head movement, update last_face_data
                            last_face_data = (face_2d, face_3d, cam_matrix, dist_matrix, curr_angles)

                    else:
                        # First iteration, update last_face_data
                        last_face_data = (face_2d, face_3d, cam_matrix, dist_matrix, [angles[0], angles[1]])

                    # Add the text on the frame
                    cv.putText(frame, text, (40, 20), cv.FONT_HERSHEY_SIMPLEX,
                               1, (0, 0, 255), 2)



            # Check if the head is back within the frame
            is_tracking = True

            if num_eyes == 2:
                # Both eyes are visible
                ratio , blink_amplitude = utils.blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

                # blink occurred
                if not initial_eye_closed and ratio > 5.5:
                    initial_eye_closed = True
                    if blink_start_time == 0:
                        
                        blink_start_time = time.time()
                        micro_sleep_start_time = 0
                        sleep_start_time = 0
                        unresponsive_start_time = 0
                        micro_sleep_detected = False
                        sleep_detected = False
                        unresponsive_detected = False
                    else:
                        # 
                        blink_duration = time.time() - blink_start_time
                        blink_times.append(time.time())
                        blink_history.append(blink_duration)

                            
                        if len(blink_history) > BLINK_HISTORY_SIZE:
                            blink_history.pop(0)

                        average_blink_duration = sum(blink_history) / len(blink_history)
                        print("Blink Duration :", average_blink_duration)
                        print("Blink Start Time:", blink_start_time)
                        print("Blink Duration:", blink_duration)
                        print("Blink History:", blink_history)
                        print("Drowsy State:", drowsy_state)


                        if average_blink_duration > BLINK_THRESHOLD:
                            
                            
                            drowsy_state = "Drowsy"
                        else:
                            drowsy_state = "Alert"
                            opening_speed = 1 / blink_duration

                        if blink_duration >= MICRO_SLEEP_DURATION_THRESHOLD and not micro_sleep_detected:
                            micro_sleep_detected = True
            

       
                        if blink_duration >= SLEEP_DURATION_THRESHOLD and not sleep_detected:
                            sleep_detected = True
           

        
                        if blink_duration >= UNRESPONSIVE_DURATION_THRESHOLD and not unresponsive_detected:
                             unresponsive_detected = True
                            
                       
                else:
                   
                    blink_start_time = 0
                    drowsy_state = "Alert"
            elif num_eyes == 1:
                
                drowsy_state = last_drowsy_state
            else:
               
                drowsy_state = "Unknown"

            # Update last drowsiness state
            if drowsy_state != "Unknown":
                last_drowsy_state = drowsy_state
            if initial_eye_closed:
                utils.colorBackgroundText(frame, f'State of Driver :  {drowsy_state}', FONTS, 0.7, (30, 400), 2)

            # Display drowsiness state
            utils.colorBackgroundText(frame, f'State of Driver :  {drowsy_state}', FONTS, 0.7, (30, 400), 2)                        

            utils.colorBackgroundText(frame, f'Opening Speed: {round(opening_speed, 2)} (1/s)', FONTS, 0.7, (30, 280), 2)

            utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

            if ratio > 5.5:
             CEF_COUNTER +=1
               
             utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
            if blink_duration > 0:
                utils.colorBackgroundText(frame, f'Blink Duration: {round(blink_duration, 2)}s', FONTS, 0.7, (30, 180), 2)
            if len(blink_times) >= 2:
                blink_duration = blink_times[-1] - blink_times[0]
                if blink_duration > 0:
                    blink_frequency = (len(blink_times) - 1) / blink_duration * 60  # 
                    
            else:
                blink_frequency = 0


            
            if CEF_COUNTER>CLOSED_EYES_FRAME:
                    TOTAL_BLINKS +=1
                    CEF_COUNTER =0

            utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
            
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            crop_right, crop_left = utils.eyesExtractor(frame, right_coords, left_coords)

            eye_pos_right, clr, direction_right = utils.positionEstimator(crop_right)
            utils.colorBackgroundText(frame, f'Right Eye: {direction_right}', FONTS, 1, (100, 500), 1, clr)

            eye_pos_left, clr, direction_left = utils.positionEstimator(crop_left)
            utils.colorBackgroundText(frame, f'Left Eye: {direction_left}', FONTS, 1, (100, 550), 1, clr)



            utils.colorBackgroundText(frame, f'Blink Frequency: {round(blink_frequency, 2)} Blinks/min', FONTS, 0.7, (30, 220), 2)
            utils.colorBackgroundText(frame, f'Blink Amplitude: {round(blink_amplitude, 2)}', FONTS, 0.7, (30, 250), 2)
            if micro_sleep_detected:
             utils.colorBackgroundText(frame, "Micro Sleep Detected!", FONTS, 1.5, (int(frame_width/2), int(frame_height/2) - 100), 2, utils.RED, pad_x=6, pad_y=6)

            if sleep_detected:
             utils.colorBackgroundText(frame, "Sleep Detected!", FONTS, 1.5, (int(frame_width/2), int(frame_height/2)), 2, utils.RED, pad_x=6, pad_y=6)

            if unresponsive_detected:
             utils.colorBackgroundText(frame, "Unresponsive Driver!", FONTS, 1.5, (int(frame_width/2), int(frame_height/2) + 100), 2, utils.RED, pad_x=6, pad_y=6)
        else:
            cv.putText(frame, "Head Out of Range", (20, 20), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 2)
            cv.putText(frame, "Re-Initializing...", (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 2)
            is_tracking = False
            reinitialize_face_tracking()

            


        # calculating  FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)

        cv.imshow('frame', frame)
        
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break

    cv.destroyAllWindows()
    camera.release()
