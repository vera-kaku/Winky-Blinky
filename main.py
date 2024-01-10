import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# test camera
cap = cv2.VideoCapture(1)
ret, img = cap.read()

while ret:
    ret, img = cap.read()

    cv2.imshow("img", img)
    a = cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
