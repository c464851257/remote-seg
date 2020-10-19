from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import cv2
from collections import  Counter
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
import utils.lovasz_softmax as L
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt



def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='F:\\datasets\\data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=9,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_res2net101_v1b',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv3plus_resnext101_32x8d','deeplabv3plus_res2net101_v1b'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=1000000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='cos', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=48,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=128,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=256)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='smooth',
                        choices=['cross_entropy', 'focal_loss','OHEM', 'lovasz_softmax',
                                 'smooth'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=2000,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser

def smooth_one_hot(true_labels, classes, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=true_labels.shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        print(true_dist.shape, true_labels.shape)
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

def open_specified_layers(model):
    """
    Open specified layers in model for training while keeping
    other layers frozen.

    Args:
    - model (nn.Module): neural net model.
    - open_layers (list): list of layer names.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    # for layer in open_layers:
    #     assert hasattr(model, layer), "'{}' is not an attribute of the model, please provide the correct name".format(layer)

    for name, module in model.named_children():
        # if name in open_layers:
        if  'classifier' in name or 'layer4' in name:
            # print('open', name)
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
            open_specified_layers(module)


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            # et.ExtRandomScale((0.8, 1.3)),
            # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtColorJitter(brightness=0.3, contrast=0, saturation=0),
            et.ExtRandomHorizontalFlip(),
            et.ExtRandomVerticalFlip(),
            et.ExtRandomRotation([0, 10]),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                # et.ExtRandomHorizontalFlip(),
                # et.ExtColorJitter(brightness=0.3, contrast=0, saturation=0),
                # et.ExtRandomRotation([0, 10]),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
    with torch.no_grad():
        for i, (images, labels, img_ids) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            # outputs = model(images)
            # # 水平翻转
            predict_2 = model(torch.flip(images, [-1]))
            predict_2 = torch.flip(predict_2, [-1])
            # 垂直翻转
            predict_3 = model(torch.flip(images, [-2]))
            predict_3 = torch.flip(predict_3, [-2])
            # 水平垂直翻转
            predict_4 = model(torch.flip(images, [-1, -2]))
            predict_4 = torch.flip(predict_4, [-1, -2])
            predict_list = outputs + predict_2 + predict_3 + predict_4
            predict_list = torch.argmax(predict_list.cpu(), 1).byte().numpy()  # n x h x w

            # for i in range(len(images)):
            #     count = Counter(preds[i].flatten())
            #     flag = -1
            #     for (k, v) in count.items():
            #         if int(v) > 0.93 * 265 * 256:
            #             flag = int(k)
            #     if flag != -1:
            #         preds[i] = np.full((256, 256), flag)
            metrics.update(targets, predict_list)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]
                    img_id = img_ids[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%s_image.png' % img_id)
                    Image.fromarray(target).save('results/%s_target.png' % img_id)
                    Image.fromarray(pred).save('results/%s_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%s_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()

        score = metrics.get_results()
    return score, ret_samples

def test(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        # img_id = 0
    matches = [100, 200, 300, 400, 500, 600, 700, 800]
    import cv2
    with torch.no_grad():
        for i, (images, labels, img_ids) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            if opts.save_val_results:
                for i in range(len(images)):
                    pred = preds[i]
                    img_id = img_ids[i]

                    pr = pred.reshape((256, 256))
                    seg_img = np.zeros((256, 256), dtype=np.uint16)
                    for c in range(8):
                        seg_img[pr[:, :] == c] = c
                    seg_img = cv2.resize(seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
                    save_img = np.zeros((256, 256), dtype=np.uint16)
                    for i in range(256):
                        for j in range(256):
                            save_img[i][j] = matches[int(seg_img[i][j])]
                    cv2.imwrite('results/%s.png' % img_id, save_img)

        score = metrics.get_results()
    return score, ret_samples

from multiprocessing import Pool

def predict(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')

    with torch.no_grad():
        # for i, (images, labels) in tqdm(enumerate(loader)):
        for i, (images, labels, img_ids) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)

            outputs = model(images)
            # 水平翻转
            predict_2 = model(torch.flip(images, [-1]))
            predict_2 = torch.flip(predict_2, [-1])
            # 垂直翻转
            predict_3 = model(torch.flip(images, [-2]))
            predict_3 = torch.flip(predict_3, [-2])
            # 水平垂直翻转
            predict_4 = model(torch.flip(images, [-1, -2]))
            predict_4 = torch.flip(predict_4, [-1, -2])
            predict_list = outputs + predict_2 + predict_3 + predict_4
            predict_list = torch.argmax(predict_list.cpu(), 1).byte().numpy()  # n x h x w
            # preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            tmp = zip(predict_list, img_ids)
            pool = Pool(4)
            pool.map(write_img_result, tmp)
            pool.close()
            pool.join()

matches = [100, 200, 300, 400, 500, 600, 700, 800]
def write_img_result(tmp):
    pred = tmp[0]
    img_id = tmp[1]
    pr = pred.reshape((256, 256))
    seg_img = np.zeros((256, 256), dtype=np.uint16)
    for c in range(8):
        seg_img[pr[:, :] == c] = c
    seg_img = cv2.resize(seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    save_img = np.zeros((256, 256), dtype=np.uint16)
    for i in range(256):
        for j in range(256):
            save_img[i][j] = matches[int(seg_img[i][j])]
    cv2.imwrite('results/%s.png' % img_id, save_img)

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 8
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 128
    
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'deeplabv3plus_resnext101_32x8d':network.deeplabv3plus_resnext101_32x8d,
        'deeplabv3plus_res2net101_v1b':network.deeplabv3plus_res2net101_v1b
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=4e-08)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'OHEM':
        criterion = utils.OhemCrossEntropy(ignore_label=255)
    elif opts.loss_type == 'smooth':
        criterion = utils.LabelSmoothSoftmaxCEV2()
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("best mean IOU %s" % best_score)
        best_score = 0.789
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        # val_score, ret_samples = validate(
        #     opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        # val_score, ret_samples = test(
        #     opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        predict(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        # print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        # open_specified_layers(model)
        for (images, labels, img_ids)   in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images)
            # smooth_label = smooth_one_hot(labels, classes=8, smoothing=0.1)
            if opts.loss_type == 'lovasz_softmax':
                outputs = F.softmax(outputs, dim=1)
                loss = L.lovasz_softmax(outputs, labels, ignore=255)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset,opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()  

            if cur_itrs >=  opts.total_itrs:
                return

        
if __name__ == '__main__':
    main()
