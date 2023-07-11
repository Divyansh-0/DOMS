
import cv2 as cv
import mediapipe as mp
import time
import utils
import numpy as np

# variables 
frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
prev_blink_time = 0


# constants
CLOSED_EYES_FRAME = 3
FONTS =cv.FONT_HERSHEY_COMPLEX
MICRO_SLEEP_DURATION_THRESHOLD = 0.5 # 
SLEEP_DURATION_THRESHOLD = 3.0  #
UNRESPONSIVE_DURATION_THRESHOLD = 4.0
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

# camera object 
camera = cv.VideoCapture(0)

with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

        # Constants for blink duration calculation
    BLINK_THRESHOLD = 0.2  # Threshold to blink (in seconds)
    BLINK_HISTORY_SIZE = 5  #

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
        #  resizing frame
        
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = utils.landmarksDetection(frame, results, False)

                         # Check the number of visible eyes
            num_eyes = 0
            if len(mesh_coords) >= len(LEFT_EYE) or len(mesh_coords) >= len(RIGHT_EYE):
                num_eyes = 2
            elif len(mesh_coords) >= len(LEFT_EYE) or len(mesh_coords) >= len(RIGHT_EYE):
                num_eyes = 1

            if num_eyes == 2:
                # Both eyes are visible
                ratio , blink_amplitude = utils.blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

                # blink occurred
                if ratio > 5.5:
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
                            # Remove oldest blink duration 
                            blink_history.pop(0)

                        average_blink_duration = sum(blink_history) / len(blink_history)


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

            # Display drowsiness state
            utils.colorBackgroundText(frame, f'Drowsiness: {drowsy_state}', FONTS, 0.7, (30, 400), 2)                        

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


            utils.colorBackgroundText(frame, f'Blink Frequency: {round(blink_frequency, 2)} Blinks/min', FONTS, 0.7, (30, 220), 2)
            utils.colorBackgroundText(frame, f'Blink Amplitude: {round(blink_amplitude, 2)}', FONTS, 0.7, (30, 250), 2)
            if micro_sleep_detected:
             utils.colorBackgroundText(frame, "Micro Sleep Detected!", FONTS, 1.5, (int(frame_width/2), int(frame_height/2) - 100), 2, utils.RED, pad_x=6, pad_y=6)

            if sleep_detected:
             utils.colorBackgroundText(frame, "Sleep Detected!", FONTS, 1.5, (int(frame_width/2), int(frame_height/2)), 2, utils.RED, pad_x=6, pad_y=6)

            if unresponsive_detected:
             utils.colorBackgroundText(frame, "Unresponsive Driver!", FONTS, 1.5, (int(frame_width/2), int(frame_height/2) + 100), 2, utils.RED, pad_x=6, pad_y=6)


            


        # calculating  FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time

        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)

        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break

    cv.destroyAllWindows()
    camera.release()
