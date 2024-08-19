import cv2
import urllib.request as urlreq
import os
import numpy as np
import time

def download_file(url, local_path):
    if not os.path.exists(local_path):
        urlreq.urlretrieve(url, local_path)
        print(f"{local_path} 已下载")
    else:
        print(f"{local_path} 已存在")

LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
LBFmodel = 'lbfmodel.yaml'
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
haarcascade = 'haarcascade_frontalface_alt2.xml'

download_file(LBFmodel_url, LBFmodel)
download_file(haarcascade_url, haarcascade)

detector = cv2.CascadeClassifier(haarcascade)

if hasattr(cv2.face, 'createFacemarkLBF'):
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)
else:
    raise ImportError("OpenCV contrib module is not available")

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

left_eye_indices = list(range(36, 42))
right_eye_indices = list(range(42, 48))

# 初始化离开计数器和时间戳
departure_count = 0
leave_start_time = None

previous_left_pupil = None
previous_right_pupil = None

def find_pupil_by_intensity(eye_region):
    if eye_region is None or eye_region.size == 0:
        return None
    
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    thresholded_eye = cv2.adaptiveThreshold(blurred_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        return (int(x), int(y))
    
    return None

def crop_eye_region(frame, landmark, eye_indices):
    x_min = int(min(landmark[eye_indices, 0]))
    x_max = int(max(landmark[eye_indices, 0]))
    y_min = int(min(landmark[eye_indices, 1]))
    y_max = int(max(landmark[eye_indices, 1]))
    
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)
    
    return frame[y_min:y_max, x_min:x_max]

def is_pupil_out_of_bounds(pupil_pos, eye_landmarks, tolerance=0.10):
    eye_width = np.linalg.norm(eye_landmarks[3] - eye_landmarks[0])
    eye_height = np.linalg.norm(eye_landmarks[4] - eye_landmarks[1])
    
    aspect_ratio = eye_height / eye_width
    adjusted_tolerance = tolerance * aspect_ratio
    
    pupil_x_ratio = (pupil_pos[0] - eye_landmarks[0][0]) / eye_width
    pupil_y_ratio = (pupil_pos[1] - eye_landmarks[1][1]) / eye_height
    
    if pupil_x_ratio < adjusted_tolerance or pupil_x_ratio > 1 - adjusted_tolerance or pupil_y_ratio < adjusted_tolerance or pupil_y_ratio > 1 - adjusted_tolerance:
        return True
    return False

def smooth_pupil_position(prev_pupil, curr_pupil, alpha=0.5):
    if prev_pupil is None:
        return curr_pupil
    return (int(alpha * prev_pupil[0] + (1 - alpha) * curr_pupil[0]), 
            int(alpha * prev_pupil[1] + (1 - alpha) * curr_pupil[1]))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(image_gray)
    
    if len(faces) > 0:
        _, landmarks = landmark_detector.fit(image_gray, faces)
        
        pupil_left_view = False
        
        for landmark in landmarks:
            for i, (x, y) in enumerate(landmark[0]):
                if i in left_eye_indices + right_eye_indices:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), 2)
                else:
                    cv2.circle(frame, (int(x), int(y)), 1, (255, 255, 255), 1)
            
            left_eye_region = crop_eye_region(frame, landmark[0], left_eye_indices)
            right_eye_region = crop_eye_region(frame, landmark[0], right_eye_indices)
            
            left_pupil = find_pupil_by_intensity(left_eye_region)
            right_pupil = find_pupil_by_intensity(right_eye_region)
            
            if left_pupil:
                left_pupil_abs = (left_pupil[0] + int(min(landmark[0][left_eye_indices, 0])), 
                                  left_pupil[1] + int(min(landmark[0][left_eye_indices, 1])))
                left_pupil_abs = smooth_pupil_position(previous_left_pupil, left_pupil_abs)
                previous_left_pupil = left_pupil_abs
                cv2.circle(frame, left_pupil_abs, 3, (0, 0, 255), -1)
                pupil_left_view = is_pupil_out_of_bounds(left_pupil_abs, landmark[0][left_eye_indices], tolerance=0.15)
            
            if right_pupil:
                right_pupil_abs = (right_pupil[0] + int(min(landmark[0][right_eye_indices, 0])), 
                                   right_pupil[1] + int(min(landmark[0][right_eye_indices, 1])))
                right_pupil_abs = smooth_pupil_position(previous_right_pupil, right_pupil_abs)
                previous_right_pupil = right_pupil_abs
                cv2.circle(frame, right_pupil_abs, 3, (0, 0, 255), -1)
                pupil_left_view = pupil_left_view or is_pupil_out_of_bounds(right_pupil_abs, landmark[0][right_eye_indices], tolerance=0.15)
        
        if pupil_left_view:
            if leave_start_time is None:
                leave_start_time = time.time()
            elif time.time() - leave_start_time > 0.5:
                departure_count += 1
                leave_start_time = None
        else:
            leave_start_time = None

    # 显示总离开次数在视频左下角
    cv2.putText(frame, f"Total Leaves: {departure_count}", (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()