import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2
import math
import random

def dwt_fuse(image1, image2, weights):
    coefficients_1 = pywt.wavedec2(image1, 'haar', level=2)
    coefficients_2 = pywt.wavedec2(image2, 'haar', level=2)

    [cA, (cH1, cV1, cD1), (cH2, cV2, cD2)] = coefficients_1
    [cAA, (cH11, cV11, cD11), (cH22, cV22, cD22)] = coefficients_2

    f_cA =  cA * weights[0] + cAA * (1 - weights[0])

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

def dwt_fuse2(image1, image2):

    coefficients_1 = pywt.wavedec2(image1, 'haar', level=2)
    coefficients_2 = pywt.wavedec2(image2, 'haar', level=2)

    coeffs = coeffs_selection(coefficients_1, coefficients_2)

    result = pywt.waverec2(coeffs, 'haar').astype('uint8')
    return result

def coeffs_selection(coefficients_1, coefficients_2):
    merged_coeffs = []
    weight = random.uniform(0., 1);

    f_cA =  coefficients_1[0] * weight + coefficients_2[0] * (1 - weight)
    coefficients_1.pop(0)
    coefficients_2.pop(0)

    merged_coeffs.append(f_cA)
    for i, coeff in enumerate(coefficients_1):
        (cH1, cV1, cD1) = coeff
        (cH2, cV2, cD2) = coefficients_2[i]
        weight = random.uniform(0., 1);

        f_cH1 = cH1 * weight + cH2 * (1 - weight)
        f_cV1 = cV1 * weight + cV2 * (1 - weight)
        f_cD1 = cD1 * weight + cD2 * (1 - weight)

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
    print(temp1[0])
    temp3[0] = (temp1[0] + temp2[0])
    temp3[1] = (temp1[1] + temp2[1])
    temp3[2] = (temp1[2] + temp2[2])
    coefficients_h[1] = tuple(temp3)
    # Creating fused image by reconstructing the fused decomposed image
    result = pywt.waverec2(coefficients_h, 'haar')
    return result

# id_x for selecting image, you can change manually 1 to 21 (21 different infrared and 21 different visible image)

def wavelet_coefficients (img, level):
    # img = cv2.imread('./images/Lena.bmp', cv2.IMREAD_GRAYSCALE)
    coeffs = pywt.wavedec2(img, 'haar', level=level)
    arr, slices = pywt.coeffs_to_array(coeffs)


    return arr, slices, coeffs


face = cv2.imread('example/face.png', 0)
finger = cv2.imread('example/finger.png', 0)
lena  = cv2.imread('example/lena.png', 0)

for i in range(10):
   cv2.imwrite("example/list["+str(i)+"].jpg", dwt_fuse2(finger, face))

r_sift = cv2.SIFT_create(250)

for i in range(10):
    face = cv2.imread('example/list[' + str(i) + '].jpg', 0)
    face_img = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    kp = r_sift.detect(face_img,None)
    face_kp =r_sift.detect(face_img,None)
    fimg = cv2.drawKeypoints(face_img,face_kp,face)
    cv2.imwrite("result_"+str(i)+".jpg", fimg)
