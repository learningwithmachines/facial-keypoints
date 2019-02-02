import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
import cupy as cp
import core.memoizers as memoize
import face_recognition as facerecog

dirpath = 'F:\\git\\aind\\deeplearn\\cv_keypoints_capstone\\'
fcascade = cv2.CascadeClassifier(dirpath + '/detector_architectures/haarcascade_frontalface_default.xml')
clahe = cv2.createCLAHE(clipLimit=1.99, tileGridSize=(8, 8))
eyecascade = cv2.CascadeClassifier(dirpath + '/detector_architectures/haarcascade_eye.xml')

dlibhogdetect = dlib.get_frontal_face_detector()
face_cnn_weights = dirpath + 'detector_architectures/mmod_human_face_detector.dat'
dlibcnndetect = dlib.cnn_face_detection_model_v1(face_cnn_weights)
pt5landmark = dirpath + 'detector_architectures/shape_predictor_5_face_landmarks.dat'
dlib5landmarks = dlib.shape_predictor(pt5landmark)

# HELPERS
def tocvrect(f: object, resizeby: int=1) -> np.ndarray:
    l, t, r, b = f.left(), f.top(), f.right(), f.bottom()
    return np.array([l, t, r - l, b - t]).astype(np.int) * resizeby


def dlib_5s_tonp(landmark_: object, resizeby: int=1) -> np.ndarray:

    mark5s = np.zeros((5, 2), dtype=np.int)

    for i in range(0, 5):
        mark5s[i] = (landmark_.part(i).x, landmark_.part(i).y)
    return mark5s * resizeby


def puttext(f: int) -> str:
    return '{} face(s) found'.format(f)
# IMAGE FUNCTIONS


# GREYSCALE Image Getter from PATH

@memoize.jlibmemo
def getgrayimage(imagepath: str) -> np.ndarray:
    return cv2.imread(imagepath, 0).astype(np.uint8)


# GREYSCALE Image --> CLAHE HISTOGRAM EQUALIZATION on Image Array

@memoize.jlibmemo
def equalizegray(imagearray: np.ndarray) -> np.ndarray:
    return clahe.apply(imagearray).astype(np.uint8)


# RGB Image Getter from PATH
@memoize.jlibmemo
def getcolorimage(imagepath: str) -> np.ndarray:
    return cv2.imread(imagepath, 1).astype(np.uint8)


# RGB Image --> CLAHE HISTOGRAM EQUALIZATION on Image Array, by CHANNEL
def equalizergb(imagearray: np.ndarray) -> np.ndarray:

    merge: list = []
    for channel_ in range(0, 3):
        merge.append(clahe.apply(imagearray[:, :, channel_]))
    return cv2.merge(merge).astype(np.uint8)


# OPENCV HAAR CASCADE based Face Extraction from PATH
def haarfacescv(imagepath: str) -> np.ndarray:

    imagegray = cv2.resize(getgrayimage(imagepath), dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    imagegray = equalizegray(imagegray)
    imagecol = getcolorimage(imagepath)
    fdetects = fcascade.detectMultiScale(imagegray, 2, 6, 0, (31, 31))

    for i in range(0, len(fdetects)):
        x, y, w, h = fdetects[i] * 2
        cv2.rectangle(imagecol, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)
    return imagecol[:, :, ::-1]


# OPENCV HAAR Features + Eye Landmarks on imagepaths

def hogfaceandeyescv(imagepath: str) -> np.ndarray:

    imagegray = cv2.resize(getgrayimage(imagepath), dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
    imagegray = equalizegray(imagegray)
    imagecol = getcolorimage(imagepath)
    fdetects = fcascade.detectMultiScale(imagegray, 1.3, 6)

    for i in range(0, len(fdetects)):
        x, y, w, h = fdetects[i]
        cx, cy, cw, ch = fdetects[i] * 2
        cv2.rectangle(imagecol, (cx, cy), (cx + cw, cy + ch), color=(0, 0, 255), thickness=3)
        eyegray = imagegray[y:y + h, x:x + w]
        eyecolor = imagecol[cy:cy + ch, cx:cx + cw]
        eyedetects = eyecascade.detectMultiScale(eyegray, 1.02, 6)

        for ei in range(0, len(eyedetects)):
            ex, ey, ew, eh = eyedetects[ei] * 2
            cv2.rectangle(eyecolor, (ex, ey), (ex + ew, ey + eh), color=(0, 255, 0), thickness=3)
    return imagecol[:, :, ::-1]


# DLIB HOG based Face Extraction from PATH

def hogfacesdlib(imagepath: str, nrescale: int = 1) -> np.ndarray:

    resizeby = 2
    color: np.ndarray = dlib.load_rgb_image(imagepath)
    gray: np.ndarray = dlib.as_grayscale(color)
    gray = dlib.resize_image(gray,
                             gray.shape[0] // resizeby,
                             gray.shape[1] // resizeby)
    fdetect: dlib.rectangles = dlibhogdetect(gray, nrescale)

    for i in range(0, len(fdetect)):
        (x, y, w, h) = tocvrect(fdetect[i]) * resizeby
        cv2.rectangle(color, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)
    return color


# face-recognition LIB's DLIB HOG based Face Extraction from PATH

def hogfacesdlibfr(imagepath: str, nrescale: int = 1) -> np.ndarray:

    resizeby = 2
    color: np.ndarray = dlib.load_rgb_image(imagepath)
    gray: np.ndarray = dlib.as_grayscale(color)
    gray = dlib.resize_image(gray, gray.shape[0] // resizeby, gray.shape[1] // resizeby)
    fdetect: dlib.rectangles = facerecog.face_locations(gray, nrescale)

    for i in range(0, len(fdetect)):
        (y, x, h, w) = np.array(fdetect[i]) * resizeby
        (x, y, w, h) = (x, y, w-x, h-y)
        cv2.rectangle(color, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)
    return color


# DLIB CNN Features + 5 Landmarks on imagepaths

def landmarksdlibcnn(imagepath: str, nrescale: int = 1) -> np.ndarray:

    resizeby: np.int = 2
    color: np.ndarray = dlib.load_rgb_image(imagepath)
    gray: np.ndarray = dlib.as_grayscale(color)
    gray: np.ndarray = dlib.resize_image(gray, gray.shape[0] // resizeby, gray.shape[1] // resizeby)
    faces: np.ndarray = np.array(dlibcnndetect(gray, nrescale))

    for i in range(0, len(faces)):
        (x, y, w, h) = tocvrect(faces[i].rect, resizeby)
        cv2.rectangle(color, (x, y), (x + w, y + h), (255, 0, 0), 4)

        landmarks = dlib5landmarks(gray, faces[i].rect)
        landmarks = dlib_5s_tonp(landmarks, resizeby)
        for ix in range(0, len(landmarks)):
            x, y = landmarks[ix]
            cv2.circle(color, (x, y), 4, (0, 255, 0), -1)

    facetext: str = puttext(len(faces))
    cv2.putText(color, facetext,
                (10, color.shape[0]//16),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 16, 16), 2, cv2.LINE_AA)
    return color


# DLIB CNN Features + 5 Landmarks on image arrays

def ilandmarksdlibcnn(imagergb: np.ndarray, nrescale: int = 1) -> np.ndarray:

    resizeby: np.int = 2
    color: np.ndarray = np.copy(imagergb)
    gray: np.ndarray = dlib.as_grayscale(color)
    gray: np.ndarray = dlib.resize_image(gray, gray.shape[0] // resizeby, gray.shape[1] // resizeby)
    faces: np.ndarray = np.array(dlibcnndetect(gray, nrescale))

    for i in range(0, len(faces)):
        (x, y, w, h) = tocvrect(faces[i].rect, resizeby)
        cv2.rectangle(color, (x, y), (x + w, y + h), (255, 0, 0), 4)
        landmarks = dlib5landmarks(gray, faces[i].rect)
        landmarks = dlib_5s_tonp(landmarks, resizeby)

        for ix in range(0, len(landmarks)):
            x, y = landmarks[ix]
            cv2.circle(color, (x, y), 4, (0, 255, 0), -1)

    facetext: str = puttext(len(faces))
    cv2.putText(color, facetext,
                (10, color.shape[0]//16),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 31, 31), 2, cv2.LINE_AA)
    return color


# thresholded canny

def thresh_canny(grayarray: np.ndarray) -> np.ndarray:

    fgray = np.copy(grayarray)
    threshval_otsu, _ = cv2.threshold(fgray, 0, 1.99, cv2.THRESH_OTSU)
    fgray = cv2.resize(fgray, dsize=(fgray.shape[1] * 2, fgray.shape[0] * 2),
                              interpolation=cv2.INTER_NEAREST)
    fgray = clahe.apply(fgray)
    lowerlim = threshval_otsu * 0.99
    upperlim = threshval_otsu * 1.0
    fgray = cv2.Canny(fgray, lowerlim, upperlim)
    fgray = cv2.morphologyEx(fgray, cv2.MORPH_CLOSE,
                                     np.ones((4, 4)),
                                     iterations=1)
    fgray = cv2.dilate(fgray, np.ones((5, 5)))
    fgray = cv2.resize(fgray, dsize=(fgray.shape[1] // 2,
                                     fgray.shape[0] // 2),
                       interpolation=cv2.INTER_NEAREST)
    return fgray


# OPENCV Blurring and Canny Edge Detection from PATH

def detectedgescv_gray(grayimage: np.ndarray)->np.ndarray:

    gray = np.copy(grayimage)
    kernel: np.ndarray = np.ones((4,4), dtype=np.float32) / 21
    gray = cv2.filter2D(gray, -1, kernel)
    gray = thresh_canny(gray)
    return gray


# DLIB CNN Features + 5 Landmarks on imagepaths

def blurfaces(imagepath: str, nrescale: int = 1) -> np.ndarray:

    resizeby: np.int = 2
    color: np.ndarray = dlib.load_rgb_image(imagepath)
    gray: np.ndarray = dlib.as_grayscale(color)
    gray: np.ndarray = dlib.resize_image(gray,
                                         gray.shape[0] // resizeby,
                                         gray.shape[1] // resizeby)

    #get faces roi boundaries
    faces: np.ndarray = np.array(dlibcnndetect(gray, nrescale))

    for i in range(0, len(faces)):
        #convert dlib faces
        (x, y, w, h) = tocvrect(faces[i].rect, resizeby)
        #make box
        cv2.rectangle(color, (x, y), (x + w, y + h), (255, 0, 0), 4)
        #blur on roi
        cv2.boxFilter(src = color[y:y+h, x:x+w],
                      dst=color[y:y+h, x:x+w],
                      ddepth=-1,ksize=(63,63),
                      anchor=(-1,-1), normalize=True,
                      borderType=cv2.BORDER_REFLECT)
    return color