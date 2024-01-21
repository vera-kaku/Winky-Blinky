import cv2
import mediapipe as mp
import numpy as np
import time
import math

# 初始化
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
last_blink_time = time.time()
BLINK_GAP = 2
blink_threshold = 3.3  # ADJUST TESTING
eyebrow_eyelid_threshold = 47
show_emoji = False  # 控制 emoji 的显示

# Euclidean distance function
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blink ratio function
def blinkRatio(landmarks, right_indices, left_indices):
    # Right eye
    rh_right = (int(landmarks[right_indices[0]].x * width), int(landmarks[right_indices[0]].y * height))
    rh_left = (int(landmarks[right_indices[1]].x * width), int(landmarks[right_indices[1]].y * height))
    rv_top = (int(landmarks[right_indices[2]].x * width), int(landmarks[right_indices[2]].y * height))
    rv_bottom = (int(landmarks[right_indices[3]].x * width), int(landmarks[right_indices[3]].y * height))

    # Left eye
    lh_right = (int(landmarks[left_indices[0]].x * width), int(landmarks[left_indices[0]].y * height))
    lh_left = (int(landmarks[left_indices[1]].x * width), int(landmarks[left_indices[1]].y * height))
    lv_top = (int(landmarks[left_indices[2]].x * width), int(landmarks[left_indices[2]].y * height))
    lv_bottom = (int(landmarks[left_indices[3]].x * width), int(landmarks[left_indices[3]].y * height))

    # Calculate distances
    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)
    lvDistance = euclideanDistance(lv_top, lv_bottom)

    # Calculate ratios
    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio

# Indices for right and left eyes
right_indices = [33, 133, 159, 145]
left_indices = [362, 263, 386, 374]


# 新的眨眼检测函数
def eyebrow_eyelid_BlinkDetection(landmarks, height, width):
    # 右眼坐标点
    right_eyebrow = (int(landmarks[52].x * width), int(landmarks[52].y * height))
    right_eyelid = (int(landmarks[159].x * width), int(landmarks[159].y * height))

    # 左眼坐标点
    left_eyebrow = (int(landmarks[295].x * width), int(landmarks[295].y * height))
    left_eyelid = (int(landmarks[386].x * width), int(landmarks[386].y * height))

    # 计算距离
    right_eye_distance = euclideanDistance(right_eyebrow, right_eyelid)
    left_eye_distance = euclideanDistance(left_eyebrow, left_eyelid)

    # 返回平均距离
    return (right_eye_distance + left_eye_distance) / 2





while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame from camera. Check camera connection.")
        break
    
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ratio = blinkRatio(face_landmarks.landmark, right_indices, left_indices)
            eyebrow_eyelid_ratio = eyebrow_eyelid_BlinkDetection(face_landmarks.landmark, height, width)
            print(eyebrow_eyelid_ratio)

        # 检查两种方法是否都检测到眨眼
            if ratio > blink_threshold and eyebrow_eyelid_ratio < eyebrow_eyelid_threshold:
                last_blink_time = time.time()
                show_emoji = False

    if time.time() - last_blink_time > BLINK_GAP and not show_emoji:
        show_emoji = True  # 显示 emoji

    print(show_emoji)
    if show_emoji:
        emoji_img = cv2.imread('winky.png', cv2.IMREAD_UNCHANGED)
        emoji_resized = cv2.resize(emoji_img, (100, int(emoji_img.shape[0] * 100/emoji_img.shape[1])), interpolation=cv2.INTER_AREA)
        x_offset = frame.shape[1] // 2 - emoji_resized.shape[1] // 2
        y_offset = frame.shape[0] // 2 - emoji_resized.shape[0] // 2
        for c in range(0, 3):
            frame[y_offset:y_offset+emoji_resized.shape[0], x_offset:x_offset+emoji_resized.shape[1], c] = \
            emoji_resized[:, :, c] * (emoji_resized[:, :, 3]/255.0) + \
            frame[y_offset:y_offset+emoji_resized.shape[0], x_offset:x_offset+emoji_resized.shape[1], c] * (1.0 - emoji_resized[:, :, 3]/255.0)


    # 绘制面部标记
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
