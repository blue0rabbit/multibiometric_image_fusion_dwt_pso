import cv2
import os
import numpy as np
import re

lbph = cv2.face.LBPHFaceRecognizer_create()


def get_img_names():
    paths = [os.path.join('samples', p) for p in os.listdir('samples') if "TRAIN" in p]

    faces = []
    ids = []

    unique_paths = paths[::5]

    for path in unique_paths:
        face_img = cv2.imread(path, 0)

        ids.append(int(re.findall('\d+', path)[0]))
        faces.append(face_img)

    return np.array(ids), faces


ids, faces = get_img_names()

print('Training has started!')

lbph.train(faces, ids)
lbph.write('classifiers/lbphClassifier.yml')

print('Finished training!')
