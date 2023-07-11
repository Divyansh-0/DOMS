import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.6)
cap = cv2.VideoCapture(0)

last_face_data = None  # Variable to store the last known position and orientation of the face
is_tracking = True  # Variable to track if the head is currently being tracked
roi_x_min, roi_y_min, roi_x_max, roi_y_max = 200, 100, 500, 400  # Example ROI coordinates
angle_threshold = 6
max_head_movement_per_frame = 6  # Maximum allowed head movement in degrees per frame

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

        # Extend the region around the last known position and orientation
        extend_factor = 1.5  # Modify this factor as needed
        extended_x_min = int(min(face_2d[:, 0]) * (1 - extend_factor))
        extended_x_max = int(max(face_2d[:, 0]) * (1 + extend_factor))
        extended_y_min = int(min(face_2d[:, 1]) * (1 - extend_factor))
        extended_y_max = int(max(face_2d[:, 1]) * (1 + extend_factor))

        # Ensure the extended region is within the image bounds
        extended_x_min = max(extended_x_min, 0)
        extended_x_max = min(extended_x_max, img_w - 1)
        extended_y_min = max(extended_y_min, 0)
        extended_y_max = min(extended_y_max, img_h - 1)

        # Update the image by cropping the extended region
        extended_image = image[extended_y_min:extended_y_max, extended_x_min:extended_x_max]

        # Process the extended region
        results = face_mesh.process(extended_image)

        if results.multi_face_landmarks:
            # Re-initialization successful
            for face_landmarks in results.multi_face_landmarks:
                # Adjust the face landmarks to the original image coordinates
                for landmark in face_landmarks.landmark:
                    landmark.x = (landmark.x * (extended_x_max - extended_x_min) + extended_x_min) / img_w
                    landmark.y = (landmark.y * (extended_y_max - extended_y_min) + extended_y_min) / img_h

                top_landmark_indices = [10, 152, 159, 145, 153, 144, 155, 151, 158]
                face_2d = []
                face_3d = []

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

                # Update the camera matrix and distance matrix
                cam_matrix[0, 2] = img_h / 2  # Update the principal point x-coordinate
                cam_matrix[1, 2] = img_w / 2  # Update the principal point y-coordinate

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the yaw and pitch angles in degrees
                yaw = angles[0] * 360
                pitch = angles[1] * 360

                # Check if the head yaw and pitch angles are within the desired range
                if abs(yaw) < angle_threshold and abs(pitch) < angle_threshold:
                    is_tracking = True
                else:
                    is_tracking = False

                break  # Only process the first detected face for re-initialization

while True:
    ret, image = cap.read()
    if not ret:
        break

    # Flip the image horizontally and vertically for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
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

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

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

            # Add the text on the image
            cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        else:
            cv2.putText(image, "Head Out of ROI", (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv2.putText(image, "Re-Initializing...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            is_tracking = False
            reinitialize_face_tracking()

            # Check if the head is back within the ROI
            roi_center_x = (roi_x_min + roi_x_max) // 2
            roi_center_y = (roi_y_min + roi_y_max) // 2
            if roi_x_min <= roi_center_x <= roi_x_max and roi_y_min <= roi_center_y <= roi_y_max:
                is_tracking = True

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

