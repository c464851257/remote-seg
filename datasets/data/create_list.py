import os

path = 'VOCdevkit/VOC2012/JPEGImages/'

img_list = open('train_aug.txt','w')
img_list_val = open('val_aug.txt','w')

imgs = os.listdir(path)
count = 0
for img in imgs:
    img2 = img[:-4]
    if True:
        img_list.write(img2 + '\n')
        print(img2)
    else:
        img_list_val.write(img2 + '\n')
    # count += 1
# for file in imgs:
#     os.rename(os.path.join(path,file),os.path.join(path,file[:-4]+".jpg"))
#     count += 1
#     print(count)
