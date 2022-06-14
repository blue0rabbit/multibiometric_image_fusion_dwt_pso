import os
import random
import re
from enum import Enum

import cv2
import pywt

WIDTH, HEIGHT = 225, 350
weights_val = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Mode(Enum):
    TRAIN = 1
    TEST = 2


def sort(e):
    return int(re.findall('\d+', e)[0])


def dwt_fuse(image1, image2, weights):
    coefficients_1 = pywt.wavedec2(image1, 'haar', level=2)
    coefficients_2 = pywt.wavedec2(image2, 'haar', level=2)

    [cA, (cH1, cV1, cD1), (cH2, cV2, cD2)] = coefficients_1
    [cAA, (cH11, cV11, cD11), (cH22, cV22, cD22)] = coefficients_2

    f_cA = cA * weights[0] + cAA * (1 - weights[0])

    f_cH1 = cH1 * weights[1] + cH11 * (1 - weights[1])
    f_cV1 = cV1 * weights[2] + cV11 * (1 - weights[2])
    f_cD1 = cD1 * weights[3] + cD11 * (1 - weights[3])

    f_cH2 = cH2 * weights[4] + cH22 * (1 - weights[4])
    f_cV2 = cV2 * weights[5] + cV22 * (1 - weights[5])
    f_cD2 = cD2 * weights[6] + cD22 * (1 - weights[6])

    coeffs = [f_cA, (f_cH1, f_cV1, f_cD1), (f_cH2, f_cV2, f_cD2)]
    # Creating fused image by reconstructing the fused decomposed image
    result = pywt.waverec2(coeffs, 'haar').astype('uint8')
    return result


def dwt_fuse2(image1, image2, weights):
    coefficients_1 = pywt.wavedec2(image1, 'haar', level=2)
    coefficients_2 = pywt.wavedec2(image2, 'haar', level=2)

    coeffs = coeffs_selection(coefficients_1, coefficients_2, weights)

    result = pywt.waverec2(coeffs, 'haar').astype('uint8')
    return result


def coeffs_selection(coefficients_1, coefficients_2, weights):
    merged_coeffs = []

    f_cA = coefficients_1[0] * weights[0] + coefficients_2[0] * (1 - weights[0])
    coefficients_1.pop(0)
    coefficients_2.pop(0)

    merged_coeffs.append(f_cA)
    for i, coeff in enumerate(coefficients_1):
        (cH1, cV1, cD1) = coeff
        (cH2, cV2, cD2) = coefficients_2[i]

        f_cH1 = cH1 * weights[0] + cH2 * (1 - weights[1])
        f_cV1 = cV1 * weights[1] + cV2 * (1 - weights[2])
        f_cD1 = cD1 * weights[2] + cD2 * (1 - weights[3])

        merged_coeffs.append((f_cH1, f_cV1, f_cD1))

    return merged_coeffs


def fuse_coefficients(coefficients_1, slices, coefficients_2, slices2):
    coefficients_1 = pywt.array_to_coeffs(coefficients_1, slices)
    coefficients_2 = pywt.array_to_coeffs(coefficients_2, slices2)

    # creating variables to be used
    coefficients_h = list(coefficients_1)
    # fusing the decomposed image data
    coefficients_h[0] = (coefficients_1[0] + coefficients_2[0]) * 0.5
    # creating variables to be used
    temp1 = list(coefficients_1[1])
    temp2 = list(coefficients_2[1])
    temp3 = list(coefficients_h[1])
    # fusing the decomposed image data
    temp3[0] = (temp1[0] + temp2[0])
    temp3[1] = (temp1[1] + temp2[1])
    temp3[2] = (temp1[2] + temp2[2])
    coefficients_h[1] = tuple(temp3)
    # Creating fused image by reconstructing the fused decomposed image
    result = pywt.waverec2(coefficients_h, 'haar')
    return result


# id_x for selecting image, you can change manually 1 to 21 (21 different infrared and 21 different visible image)

def wavelet_coefficients(img, level):
    # img = cv2.imread('./images/Lena.bmp', cv2.IMREAD_GRAYSCALE)
    coeffs = pywt.wavedec2(img, 'haar', level=level)
    arr, slices = pywt.coeffs_to_array(coeffs)

    return arr, slices, coeffs


def create_dwt_samples(mode, weights):
    paths = [os.path.join('faces', p) for p in os.listdir('faces')]
    select_index = int(mode.value - 1)
    unique_face_paths = paths[select_index::2]

    fingers = [os.path.join('finger', p) for p in os.listdir('finger')]
    fingers.sort(key=sort)

    file = "\\p" + str(mode.value) + ".bmp"
    filee = "\\p" + str(mode.value + 2) + ".bmp"
    fileee = "\\p" + str(mode.value + 4) + ".bmp"

    unique_finger_paths = [(p + file) for p in fingers]
    unique_finger_paths_2 = [(p + filee) for p in fingers]
    unique_finger_paths_3 = [(p + fileee) for p in fingers]

    all_fingers = unique_finger_paths + unique_finger_paths_2 + unique_finger_paths_3
    all_fingers.sort(key=sort)

    # for face in unique_face_paths:
    #     for finger in unique_finger_paths:
    for i in range(0, len(all_fingers)):
        face_img = cv2.imread(unique_face_paths[i], 0)
        finger_img = cv2.imread(all_fingers[i], 0)

        img_face = cv2.resize(face_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        img_finger = cv2.rotate(finger_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imwrite("samples/sample_" + str(mode.name)
                    + "[" + (re.findall('\d+', unique_face_paths[i])[0]) + "].jpg",
                    dwt_fuse2(img_face, img_finger, weights))


def create_samples():
    weights = [random.choice(weights_val), random.choice(weights_val), random.choice(weights_val),
               random.choice(weights_val)]
    print("Create samples with weights: " + str(weights))
    create_dwt_samples(Mode.TRAIN, weights)
    create_dwt_samples(Mode.TEST, weights)
    print("Samples created!")

    return weights


# create_samples()
