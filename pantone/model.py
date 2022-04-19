"""Test file"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

"""
class colorIdentifier:

    def __init__(self, ):
        self.clase =
        self.image =
        self.color =
        self.box =
"""


def get_objs_ROI(objetos, frame):
    """Get objects ROI

    Given the output of the object detector
    It looks for the specified classes (H-helmet, V-vest),
    if the objects belong to this classes
    """
    # does not return ROI, better get ROI in main
    clases_list = []
    boxes = []
    for idx, clase in enumerate(objetos["clases"]):
        if clase == "H" or clase == "V":
            clases_list.append(clase)
            x1, y1, x2, y2 = objetos["coord"][idx]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            # helmet_frame = frame[y1:y2, x1:x2]
            # helmet_frame = cv2.resize(helmet_frame, None, fx=0.3, fy=0.3)
            boxes.append([x1, y1, x2, y2])
            # cv2.imshow("helmet", helmet_frame)

    # return helmet_frame, clase, [x1, y1, x2, y2] #must be lists
    return clases_list, boxes


def get_colorname(hsv_value):
    # Rosa no está bien (da negro)
    """Real            in opencv

    H (0 - 360) ---> (0 - 179)
    Color       H       S       V
    Azul    190-255  30-100  30-100
    Verde    90-170  30-100  30-100
    Blanco    0-360   0-20   90-100
    Negro     0-360   0-100   0-20
    Amarillo 45-70   30-100  30-100
    Rosa    290-335  30-100  30-100
    Rojo      0-30   30-100  30-100
    Rojo    350-360  30-100  30-100
    """
    """
    azul = [94, 126, 30, 100, 30, 100]
    verde = [45, 84, 30, 100, 30, 100]
    blanco = [0, 179, 0, 20, 90, 100]
    negro = [0, 179, 0, 100, 0, 20]
    amarillo = [22, 34, 30, 100, 30, 100]
    rosa = [144, 166, 30, 100, 30, 100]
    rojo_1 = [0, 15, 30, 100, 30, 100]
    rojo_2 = [174, 179, 30, 100, 30, 100]
    """
    color = ""
    h = hsv_value[0]
    s = hsv_value[1]
    if s >= 0.0 and s <= 20.0:
        color = "blanco"
    elif s >= 21.0 and s <= 255.0:
        if h >= 0 and h <= 10:
            color = "rojo"
        elif h >= 17 and h <= 43:
            color = "amarillo"
        elif h >= 44 and h <= 80:
            color = "verde"
        elif h >= 81 and h <= 126:
            color = "azul"
        elif h >= 140 and h <= 169:
            color = "rosa"
        elif h >= 170 and h <= 179:
            color = "rojo"
        else:
            color = "negro"

    return color


def get_color_HSV(hs):
    """Get color in HSV and color name

    Get predominant color in hsv colorspace and calls get_colorname
    in order to return corresponding string
    """
    # ROI = cv2.resize(ROI, None, fx = 0.3, fy = 0.3)
    # ROI = cv2.resize(ROI, (100, 100))

    # hsv_im = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)

    """ hs = []
    for line in hsv_im:
        for pixel in line:
            temp_h, temp_s, temp_v = pixel
            hs.append([temp_h, temp_s]) """

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(hs)

    list = kmeans.labels_.tolist()
    moda = max(list, key=list.count)

    row_ind = kmeans.cluster_centers_[moda, 0]
    col_ind = kmeans.cluster_centers_[moda, 1]

    hsv_value = [row_ind, col_ind, 255]
    print("HSV value ", hsv_value)

    color_name = get_colorname(hsv_value)
    print("color name ", color_name)

    """
    rgb_value = cv2.cvtColor(np.uint8([[[row_ind, col_ind, 255]]]),cv2.COLOR_HSV2BGR)

    print("rgb value ", rgb_value)

    w = x2 - x1
    h = y2 - y1
    rgb_value = (int(rgb_value[0][0][0]), int(rgb_value[0][0][1]),
    int(rgb_value[0][0][2]))

    cv2.rectangle(objetos['image'], (x1, y1), (x1+w, y1+h), rgb_value, 3)
    """

    return hsv_value, color_name  # rgb_value #, color_id


def identify_color(frame, is_drawing=False):
    """Identify color

    The predominant color in HSV colorspace is obtained
    and the colorname too
    Then that value is converted into rgb colorspace
    Finally, the object detected is enclosed in a rectangle with
    the color detected and the colorname is displayed in the image
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2

    img = cv2.resize(frame, None, fx=0.1, fy=0.1)

    hsv_im = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hs = []
    for line in hsv_im:
        for pixel in line:
            temp_h, temp_s, temp_v = pixel
            hs.append([temp_h, temp_s])

    hsv_color, color_name = get_color_HSV(hs)
    print("color name inside ", color_name)
    # cv2.imshow("frame", frame)
    # cv2.waitKey(0)
    rgb_value = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)
    # print("rgb value ", rgb_value)
    # x1, y1, x2, y2 = boxes[i]
    # w = x2 - x1
    # h = y2 - y1
    rgb_value = (
        int(rgb_value[0][0][0]),
        int(rgb_value[0][0][1]),
        int(rgb_value[0][0][2]),
    )

    if is_drawing:
        # cv2.rectangle(objetos["image"], (x1, y1), (x1 + w, y1 + h), rgb_value, 3)
        x1 = 50
        y1 = 50
        org = (x1 - 5, y1 - 5)
        cv2.putText(
            frame,
            color_name,
            org,
            font,
            fontScale,
            rgb_value,
            thickness,
            cv2.LINE_AA,
        )

    # colors_list.append(color_name)
    print("color name ", color_name)
    return color_name  # boxes, clases_list, colors_list


def identify_colors(frame, objetos, is_drawing=False):
    """Identify color

    First get ROI of all the detected objects that belong to
    classes helmet or vest.
    Then for each object detected, in the corresponding ROI,
    the predominant color in HSV colorspace is obtained
    and the colorname too
    Then that value is converted into rgb colorspace
    Finally, the object detected is enclosed in a rectangle with
    the color detected and the colorname is displayed in the image
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2

    clases_list, boxes = get_objs_ROI(objetos, frame)

    colors_list = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        ROI_frame = frame[y1:y2, x1:x2]
        # ROI_frame = cv2.resize(ROI_frame, None, fx=0.3, fy=0.3)
        # ROI_frame = cv2.resize(ROI_frame, (100, 100))

        hsv_im = cv2.cvtColor(ROI_frame, cv2.COLOR_BGR2HSV)
        hs = []
        for line in hsv_im:
            for pixel in line:
                temp_h, temp_s, temp_v = pixel
                hs.append([temp_h, temp_s])

        # for i, ROI_frame in enumerate(ROI_frames):
        # print("getting color from ROI")
        hsv_color, color_name = get_color_HSV(hs)  # ROI_frame)
        if color_name != "amarillo":
            cv2.imshow("roi", ROI_frame)
        rgb_value = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)
        # print("rgb value ", rgb_value)
        x1, y1, x2, y2 = boxes[i]
        w = x2 - x1
        h = y2 - y1
        rgb_value = (
            int(rgb_value[0][0][0]),
            int(rgb_value[0][0][1]),
            int(rgb_value[0][0][2]),
        )

        if is_drawing:
            cv2.rectangle(objetos["image"], (x1, y1), (x1 + w, y1 + h), rgb_value, 3)
            org = (x1 - 5, y1 - 5)
            cv2.putText(
                objetos["image"],
                color_name,
                org,
                font,
                fontScale,
                rgb_value,
                thickness,
                cv2.LINE_AA,
            )

        colors_list.append(color_name)

    return boxes, clases_list, colors_list
