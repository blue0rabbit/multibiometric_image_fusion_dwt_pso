from skimage import feature
import numpy as np
import cv2
import os
import json
import re
from scipy.spatial import distance
from scipy.spatial import procrustes

histograms = []
r_sift = cv2.SIFT_create()


class HistData:
    def __init__(self, id, hist):
        self.id = id
        self.hist = hist


class SiftData:
    def __init__(self, id, sift):
        self.id = id
        self.sift = sift


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def match(img1_hist, img2):
    best_score = 0.02
    best_name = None
    n_bins = 64 + 1
    # img1_hist, _ = np.histogram(lbp1, normed=True, bins=n_bins, range=(0, n_bins))
    # img2_hist, _ = np.histogram(lbp2, normed=True, bins=n_bins, range=(0, n_bins))
    img2_hist, _ = getLBPH(img2)
    score = kullback_leibler_divergence(img2_hist, img1_hist)
    # print(score)
    return score


def compare_desc(desc1, desc2):
    bf = cv2.BFMatcher()
    # Match descriptors.
    # matches = bf.match(desc1, desc2)
    matches = bf.match(desc1, desc2)

    return matches


def compareHist(hist1, hist2):
    # chi = math.ChiSquare(hist1, hist2)
    euc = distance.euclidean(hist1, hist2)
    # score = kullback_leibler_divergence(hist1, hist2)
    # matrix1, matrix2, disparity = procrustes(np.array(hist1), np.array(hist2))

    return euc


def getLBPH(image, eps=1e-7):
    # compute the LBP representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, 64, 4, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=64, range=(0, 64))

    # normalize the histogram
    # hist = hist.astype("float")
    # hist /= (hist.sum() + eps)

    # return the LBPH
    return hist, lbp


def train():
    with open('train/data.json', 'w') as outfile:
        outfile.write("")

    paths = [os.path.join('samples', p) for p in os.listdir('samples') if "TRAIN" in p]

    for path in paths:
        # print(path)
        label = (re.findall('\d+', path)[0])
        # print label
        img = cv2.imread(path, 0)
        hist_lbp, _ = getLBPH(img)
        save_train_data(hist_lbp, label, "hist")


def train_sift():
    with open('train/data.json', 'w') as outfile:
        outfile.write("")

    paths = [os.path.join('samples', p) for p in os.listdir('samples') if "TEST" in p]

    for path in paths:
        # print(path)
        label = (re.findall('\d+', path)[0])
        # print label
        img = cv2.imread(path, 0)

        kp, descriptor = r_sift.detectAndCompute(img, None)
        save_train_data(descriptor, label, "sift")


def save_train_data(sift_data, label, name):
    with open('train/data.json', 'a') as outfile:
        data = {}
        data['id'] = label

        data[name] = sift_data.tolist()
        histograms.append(HistData(data["id"], sift_data))
        json.dump(data, outfile)
        outfile.write('\n')
        outfile.close()


def load_train_file():
    filename = 'train/data.json'

    for line in open(filename, mode="r"):
        var = json.loads(line)
        histograms.append(HistData(var["id"], (var["hist"])))
    # return var["id"], var["hist"]
    # print(var["id"])
    # print(var["hist"])


def calculate_results(id1, id2):
    if str(int(id1))[:-1] == str(int(id2))[:-1]:
        print(str(int(id1)) + " - " + str(int(id2)))
        print("Equal")
        return 1
    else:
        return 0


def lbph_verification_tests():
    load_train_file()

    paths = [os.path.join('samples', p) for p in os.listdir('samples') if "TEST" in p]
    lLabel = -1
    result  = 0

    for path in paths:
        img = cv2.imread(path, 0)

        test_hist, lbp = getLBPH(img)

        lDiff = 1000
        for h in histograms:
            tempDiff = compareHist(h.hist, test_hist)
            tempLabel = h.id

            if tempDiff <= lDiff:
                lDiff = tempDiff
                lLabel = tempLabel

            elif lDiff > 100000:
                lLabel = "Unknown!"
        result = result + calculate_results(re.findall('\d+', path)[0], lLabel)
    return result


face = cv2.imread('samples\\sample_TEST[0002].jpg', 0)
face2 = cv2.imread('samples\\sample_TEST[0004].jpg', 0)
# finger = cv2.imread('finger\\p29\\p1.bmp', 0)
# lena = cv2.imread('example/lena.png', 0)

# train_sift()
#
# train()

lLabel = -1
result = 0

# for path in paths:
#     img = cv2.imread(path, 0)
#
#     kp, desc = r_sift.detectAndCompute(img, None)
#
#     lDiff = 10000
#     for h in histograms:
#         matches = compare_desc(h.sift, desc)
#         good = []
#         for m, n in matches:
#             if m.distance < n.distance:
#                 good.append(m)
#                 print("Legit " + str(h.id))
#         tempLabel = h.id
#
#         # if tempDiff <= lDiff:
#         #     lDiff = tempDiff
#         #     lLabel = tempLabel
#
#         # elif lDiff > 0.1:
#         #     lLabel = "Unknown!"
#     result = result + calculate_results(re.findall('\d+', path)[0], lLabel)

# train()
cv2.waitKey(0)
