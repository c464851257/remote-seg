新建一个文件夹checkpoints

下载训练好的权重放进checkpoints

下载地址：

链接：https://pan.baidu.com/s/1-1PRowBbLRx0LVNCWbVbFA 
提取码：nx3w 

需要推理的图片放进 datasets/data/VOCdevkit/VOC2012/JPEGImages下面，并且新建一个文件夹SegmentationClassAug与JPEGImages在相同目录

cd datasets/data/

运行create_list.py文件会生成一个test.txt文件，记录所有要推理图片的文件名

到根目录下运行命令

python main.py --model deeplabv3plus_res2net101_v1b --gpu_id 0 --year 2012_aug --save_val_results --ckpt checkpoints/best_deeplabv3plus_res2net101_v1b_voc_os16.pth --test_only

会产生推理结果到results文件下
