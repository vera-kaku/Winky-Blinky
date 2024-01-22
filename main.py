import cv2
import mediapipe as mp
import numpy as np
import time
import math

BLINK_GAP = 2
DEFAULT_EYE_BLINK_THRESHOLD = 3

LEFT_EYE_RATIO_CACHE = []
RIGHT_EYE_RATIO_CACHE = []
LAST_BLINK = []
SHOW_EMOJI = []

# Initialization
# cap = cv2.VideoCapture("test1.MOV")
# cap = cv2.VideoCapture("test2.MOV")
# cap = cv2.VideoCapture("test3.MOV")
cap = cv2.VideoCapture(1)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=3)


# Euclidean distance function
def euclidean_distance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

# Display winky emoji on nose tip
def display_emoji(frame, landmark):
    emoji_img = cv2.imread("winky.png", cv2.IMREAD_UNCHANGED)
    emoji_resized = cv2.resize(
        emoji_img,
        (100, int(emoji_img.shape[0] * 100 / emoji_img.shape[1])),
        interpolation=cv2.INTER_AREA,
    )
    # Nose tip is landmark 4 based on MediaPipe's face landmark model
    x_offset = int(landmark[4].x * frame.shape[1]) - emoji_resized.shape[1] // 2
    y_offset = int(landmark[4].y * frame.shape[0]) - emoji_resized.shape[0] // 2

    for c in range(0, 3):
        frame[
            y_offset : y_offset + emoji_resized.shape[0],
            x_offset : x_offset + emoji_resized.shape[1],
            c,
        ] = emoji_resized[:, :, c] * (emoji_resized[:, :, 3] / 255.0) + frame[
            y_offset : y_offset + emoji_resized.shape[0],
            x_offset : x_offset + emoji_resized.shape[1],
            c,
        ] * (
            1.0 - emoji_resized[:, :, 3] / 255.0
        )
    return frame

# Draw heartbeat graph of blink ratio
def draw_heartbeat(data, width, height):
    graph = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(1, len(data)):
        # Normalize data (range 2-4) to fit in the graph
        # First, shift down by 2, then scale by dividing by 2 (the new max)
        # normalized_y1 = (data[i - 1] - 2) / 2
        # normalized_y2 = (data[i] - 2) / 2
        normalized_y1 = (data[i - 1] - 1) / 4
        normalized_y2 = (data[i] - 1) / 4

        x1, y1 = (i - 1) * 2, height - int(normalized_y1 * height)
        x2, y2 = i * 2, height - int(normalized_y2 * height)
        cv2.line(graph, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return graph

# Blink ratio function
def eye_blink_detection(idx, landmarks):
    # Right eye
    rh_right = (
        int(landmarks[33].x * width),
        int(landmarks[33].y * height),
    )  # right eye right idx 33
    rh_left = (
        int(landmarks[133].x * width),
        int(landmarks[133].y * height),
    )  # right eye left idx 133
    rv_top = (
        int(landmarks[159].x * width),
        int(landmarks[159].y * height),
    )  # right eye top idx 159
    rv_bottom = (
        int(landmarks[145].x * width),
        int(landmarks[145].y * height),
    )  # right eye bottom idx 145

    # Left eye
    lh_right = (
        int(landmarks[362].x * width),
        int(landmarks[362].y * height),
    )  # left eye right idx 362
    lh_left = (
        int(landmarks[263].x * width),
        int(landmarks[263].y * height),
    )  # left eye left idx 263
    lv_top = (
        int(landmarks[386].x * width),
        int(landmarks[386].y * height),
    )  # left eye top idx 386
    lv_bottom = (
        int(landmarks[374].x * width),
        int(landmarks[374].y * height),
    )  # left eye bottom idx 374

    # Calculate distances
    rhDistance = euclidean_distance(rh_right, rh_left)
    rvDistance = euclidean_distance(rv_top, rv_bottom)
    lhDistance = euclidean_distance(lh_right, lh_left)
    lvDistance = euclidean_distance(lv_top, lv_bottom)

    # Calculate horizontal over vertical ratios
    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    # Real-time tracking of eye ratio changes
    if len(LEFT_EYE_RATIO_CACHE) < idx + 1:
        LEFT_EYE_RATIO_CACHE.append([])
    if len(RIGHT_EYE_RATIO_CACHE) < idx + 1:
        RIGHT_EYE_RATIO_CACHE.append([])

    if len(LEFT_EYE_RATIO_CACHE[idx]) >= 100:
        LEFT_EYE_RATIO_CACHE[idx].pop(0)
    LEFT_EYE_RATIO_CACHE[idx].append(leRatio)
    if len(RIGHT_EYE_RATIO_CACHE[idx]) >= 100:
        RIGHT_EYE_RATIO_CACHE[idx].pop(0)
    RIGHT_EYE_RATIO_CACHE[idx].append(reRatio)
    
    # Calculate personalized blink threshold
    average_left_ratio = sum(LEFT_EYE_RATIO_CACHE[idx][-20:])/20
    average_right_ratio = sum(RIGHT_EYE_RATIO_CACHE[idx][-20:])/20
    if len(LEFT_EYE_RATIO_CACHE[idx]) == 100 and len(RIGHT_EYE_RATIO_CACHE[idx]) == 100:
        threshold = (average_left_ratio + average_right_ratio) / 2 * 1.1
    else:
        threshold = DEFAULT_EYE_BLINK_THRESHOLD

    return threshold, leRatio, reRatio

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame from camera. Check camera connection.")
        break

    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        idx = 0
        for face_landmarks in results.multi_face_landmarks:
            eye_threshold, left_ratio, right_ratio = eye_blink_detection(idx, face_landmarks.landmark)
            eye_ratio = (left_ratio + right_ratio) / 2

            offset = idx * 300
            cv2.putText(frame, f"Left Ratio: {left_ratio:.2f}", (250, 60 + offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
            left_eye_graph = draw_heartbeat(LEFT_EYE_RATIO_CACHE[idx], 200, 100)
            cv2.putText(frame, f"Right Ratio: {right_ratio:.2f}", (250, 160 + offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
            right_eye_graph = draw_heartbeat(RIGHT_EYE_RATIO_CACHE[idx], 200, 100)
            frame[0 + offset:100 + offset, 0:200] = left_eye_graph
            frame[100 + offset:200 + offset, 0:200] = right_eye_graph

            if len(LAST_BLINK) < idx + 1:
                LAST_BLINK.append(time.time())
            if len(SHOW_EMOJI) < idx + 1:
                SHOW_EMOJI.append(False)

            # if eye_ratio > eye_threshold and eyebrow_ratio < eyebrow_threshold:
            if eye_ratio > eye_threshold:
                LAST_BLINK[idx] = time.time()
                SHOW_EMOJI[idx] = False

            since_last_blink = time.time() - LAST_BLINK[idx]

            if since_last_blink > BLINK_GAP and not SHOW_EMOJI[idx]:
                SHOW_EMOJI[idx] = True
                
            if SHOW_EMOJI[idx]:
                frame = display_emoji(frame, face_landmarks.landmark)

            if len(LEFT_EYE_RATIO_CACHE[idx]) >= 100 or len(RIGHT_EYE_RATIO_CACHE[idx]) >= 100:            
                cv2.putText(frame, f"Since Last Blink: {since_last_blink:.2f}s", (10, 260 + idx * 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
            else:
                cv2.putText(frame, f"Since Last Blink: Calibrating...", (10, 260 + idx * 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

            # Draw face contours
            # mp.solutions.drawing_utils.draw_landmarks(
            #     frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS
            # )

            idx += 1

    cv2.putText(frame, f"Blink Gap: {BLINK_GAP}s", (frame.shape[1] - 460, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
