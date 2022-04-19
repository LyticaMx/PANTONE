"""Test file"""
import cv2

from pantone.model import identify_color

frame = cv2.imread("casco.jpeg")


detected_color = identify_color(frame, is_drawing=True)
print("detected color ", detected_color)

resultado = cv2.resize(frame, None, fx=0.7, fy=0.7)
cv2.imshow("processed frame", resultado)
cv2.waitKey(0)
