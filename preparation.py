import cv2
import os
import re
from collections import OrderedDict

WIDTH, HEIGHT, FONT, SUB_FONT = 220, 220, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL

paths = [os.path.join('faces', p) for p in os.listdir('faces')]
vals = [1, 2, 3, 4, 5, 6]
to_remove = []

for i in range(0, len(paths)):
    # for path in unique_paths:
    head, tail = os.path.split(paths[i])
    file = paths[i]
    index = int(re.findall('\d+', paths[i])[0])
    for j in range(1, 7):
        new_path = str(j).join(file.rsplit(file[len(file) - (len(file) + 5):], 1)) + ".jpg"
        # print("new path: " + new_path)
        if not os.path.exists(new_path):
            to_remove.append(new_path)

result = list(OrderedDict.fromkeys(to_remove))
indexes = []

for v in result:
    # print(v[len(v) - (len(v) + 6)])
    index = (re.findall('\d+', v)[0])[:-1]
    print(int(index))
    indexes.append(index)

for index in indexes:
    for j in range(1, 7):
        path = '/faces/sub' + str(index) + str(j) + ".jpg"
        if os.path.exists(path):
            os.remove(path)

