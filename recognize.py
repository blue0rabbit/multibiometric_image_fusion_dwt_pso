import cv2
import pandas as pd
import os
import re

WIDTH, HEIGHT = 220, 220


def calculate_results(id1, id2):
    print(str(int(id1)) + " - " + str(int(id2)))
    if str(int(id1))[:-1] == str(int(id2))[:-1]:
        print("Equal")
        return 1
    else:
        return 0


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('classifiers/lbphClassifier.yml')

result = 0

paths = [os.path.join('samples', p) for p in os.listdir('samples') if "TEST" in p]
i = 0
for path in paths:
    img = cv2.imread(path, 0)
    id, trust = face_recognizer.predict(img)

    # print(path + " - " + str(id))
    result = result + int(calculate_results(id, (re.findall('\d+', path)[0])))
    #     img = cv2.imread(path)
    #     img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     detected_faces = face_detector.detectMultiScale(img_grey, scaleFactor=1.1)
    #     for x, y, w, h in detected_faces:
    #         img_face = cv2.resize(img_grey[y:y + h, x:x + w], (WIDTH, HEIGHT))
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #         i = i+1
    #         print(i)
    #         cv2.imshow("lbph.py", img)
    #         cv2.waitKey(0)
    #
    #         id, trust = face_recognizer.predict(img_face)
    #         if id != -1:
    #             try:
    #                 cv2.putText(img, id_names[id_names['id'] == id].iloc[0]['name'], (x, y + h + 30), FONT, 1, (0, 0, 255))
    #                 cv2.putText(img, str(trust), (x, y + h + 60), FONT, 0.5, (0, 0, 255))
    #             except:
    #                 pass
    # print(i)

print(result)
# id, trust = face_recognizer.predict(face_img)
