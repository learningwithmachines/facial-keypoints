import dlib
import cv2
import numpy as np
from numba.types import string, uint8


dirpath = 'F:\\git\\aind\\deeplearn\\cv_keypoints_capstone\\'
fcascade = cv2.CascadeClassifier(dirpath + '/detector_architectures/haarcascade_frontalface_default.xml')
eyecascade = cv2.CascadeClassifier(dirpath + '/detector_architectures/haarcascade_eye.xml')

clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
dlibdetect = dlib.get_frontal_face_detector()
face_cnn_weights = dirpath + 'detector_architectures/mmod_human_face_detector.dat'
dlibcnndetect = dlib.cnn_face_detection_model_v1(face_cnn_weights)

pt68landmark = dirpath + 'detector_architectures/shape_predictor_68_face_landmarks.dat'
dlib68landmarks = dlib.shape_predictor(pt68landmark)

# HELPERS


def tocvrect(f: object, resizeby: int=1) -> np.ndarray:
    l, t, r, b = f.left(), f.top(), f.right(), f.bottom()
    return np.array([l, t, r - l, b - t]).astype(np.int) * resizeby


def dlib_68s_tonp(landmark_: object, resizeby: int=1) -> np.ndarray:
    mark5s = np.zeros((68, 2), dtype=np.int)
    for i in range(0, 68):
        mark5s[i] = (landmark_.part(i).x, landmark_.part(i).y)
    return mark5s * resizeby


def puttext(f: uint8) -> string:
    return '{} face(s) found'.format(f)


# DLIB CNN Features + 68 Landmarks on image arrays
def vidmarksdlibcnn(imagergb: np.ndarray, nrescale: int = 1) -> np.ndarray:
    resizeby: np.int = 2
    color: np.ndarray = imagergb
    gray: np.ndarray = dlib.as_grayscale(color)
    gray: np.ndarray = dlib.resize_image(gray, gray.shape[0] // resizeby, gray.shape[1] // resizeby)
    faces: np.ndarray = np.array(dlibcnndetect(gray, nrescale))
    for i in range(0, len(faces)):
        (x, y, w, h) = tocvrect(faces[i].rect, resizeby)
        cv2.rectangle(color, (x, y), (x + w, y + h), (0, 0, 255), 4)

        landmarks = dlib68landmarks(gray, faces[i].rect)
        landmarks = dlib_68s_tonp(landmarks, resizeby)
        for ix in range(0, len(landmarks)):
            x, y = landmarks[ix]
            cv2.circle(color, (x, y), 4, (0, 255, 0), -1)

    facetext: str = puttext(len(faces))
    cv2.putText(color, facetext,
                (10, color.shape[0]//16),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 16, 16), 2, cv2.LINE_AA)
    return color


def vidblurfaces(imagergb: np.ndarray, nrescale: int = 1) -> np.ndarray:
    resizeby: np.int = 2
    color: np.ndarray = imagergb
    gray: np.ndarray = dlib.as_grayscale(color)
    gray: np.ndarray = dlib.resize_image(gray, gray.shape[0] // resizeby, gray.shape[1] // resizeby)
    faces: np.ndarray = np.array(dlibcnndetect(gray, nrescale))
    for i in range(0, len(faces)):
        (x, y, w, h) = tocvrect(faces[i].rect, resizeby)
        cv2.boxFilter(src = color[y:y+h, x:x+w],
                      dst=color[y:y+h, x:x+w],
                      ddepth=-1,ksize=(63,63),
                      anchor=(-1,-1), normalize=True,
                      borderType=cv2.BORDER_REFLECT)
    return color

