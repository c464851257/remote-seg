import os
import cv2
path = 'results/'

imgs = os.listdir(path)

for img in imgs:
    img = os.path.join(path,'56.png')
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    print(image.shape)
