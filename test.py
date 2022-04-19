"""Test file"""
import cv2

# from camera.model import Camera
from pantone.model import identify_color

frame = cv2.imread("coca2.jpeg")  # "casco.jpeg")
# cv2.imshow("frame", frame)
# cv2.waitKey(15)

detected_color = identify_color(frame, is_drawing=True)
print("detected color ", detected_color)

resultado = cv2.resize(frame, None, fx=0.7, fy=0.7)
cv2.imshow("processed frame", resultado)
cv2.waitKey(0)

# while True:
# if cv2.waitKey(1) == ord("q"):
#    break
