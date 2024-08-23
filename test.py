import jittor as jt
import ipdb
import numpy as np
import jclip as clip
from jittor import nn
from collections import OrderedDict
from beit_finetune import *
from functools import partial
from PIL import Image, ImageEnhance, ImageOps
import os
from tqdm import tqdm
import argparse
from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, resize, RandomHorizontalFlip, ColorJitter, RandomCrop
import random
from jittor.dataset import Dataset, DataLoader
from jittor import Module
from jittor import transform
import os

# import torch
random.seed(1)
# jt.Module.mpi_param_broadcast(root=0)

clipmodel, preprocess = clip.load("ViT-B-32.pkl")
classes = open('../Dataset/classes_b.txt').read().splitlines()

# remove the prefix Animal, Thu-dog, Caltech-101, Food-101

new_classes = []
for c in classes:
    c = c.split(' ')[0]
    if c.startswith('Animal'):
        c = c[7:]
    if c.startswith('Thu-dog'):
        c = c[8:]
    if c.startswith('Caltech-101'):
        c = c[12:]
    if c.startswith('Food-101'):
        c = c[9:]
    if c.startswith('Stanford-Cars'):
        c = c[14:]
    c = 'a photo of ' + c
    new_classes.append(c)
# ipdb.set_trace()
text = clip.tokenize(new_classes)
text_features = clipmodel.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)

clipmodel, preprocess = clip.load("ViT-B-32.pkl")
classes = open('../Dataset/classes_b.txt').read().splitlines()

# remove the prefix Animal, Thu-dog, Caltech-101, Food-101




jt.flags.use_cuda = 1
def accuracy(output, target, topk=(1,)):
    
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = (pred == target.contiguous().view(1, -1).expand_as(pred))
 
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)
    return res

def adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate, decay_epoch):
    """Sets the learning rate to the initial LR decayed by decay_rate every decay_epoch epochs"""
    lr = initial_lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = jt.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class RandomErasing:
    def __init__(self, probability=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)):
        self.probability = probability
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if random.random() < self.probability:
            img_np = np.array(img)
            img_h, img_w, _ = img_np.shape
            area = img_h * img_w

            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img_w and h < img_h:
                x1 = random.randint(0, img_h - h)
                y1 = random.randint(0, img_w - w)

                img_np[x1:x1 + h, y1:y1 + w, :] = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

            return Image.fromarray(img_np)
        return img

class Resize:
    def __init__(self, size, mode=Image.BILINEAR):
        if isinstance(size, int):
            self.size = size
        else:
            self.size = _setup_size(
                size,
                error_msg="If size is a sequence, it should have 2 values")
        self.mode = mode

    def __call__(self, img: Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        if isinstance(self.size, int):
            w, h = img.size

            short, long = (w, h) if w <= h else (h, w)
            if short == self.size:
                return img

            new_short, new_long = self.size, int(self.size * long / short)
            new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
            size = (new_h, new_w)
        return resize(img, size, self.mode)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def to_tensor(data):
    return jt.Var(data)

class ImageToTensor(object):
    def __call__(self, input):
        input = np.asarray(input)
        if len(input.shape) < 3:
            input = np.expand_dims(input, -1)
        return to_tensor(input)

def transform_(n_px):
    return Compose([
        Resize(n_px, mode=Image.BICUBIC),
        CenterCrop(n_px), _convert_image_to_rgb,
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        ImageToTensor()
    ])

class ImagePadding:
    def __init__(self, padding=4, fill=0):
        self.padding = padding
        self.fill = fill

    def __call__(self, img):
        return ImageOps.expand(img, border=self.padding, fill=self.fill)

state_dict_pretrained = jt.load('beit_cls_model_B.pkl')
cls_model = VisionTransformer(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, init_values=0.1, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
cls_model.load_parameters(state_dict_pretrained)
cls_model.reset_classifier(375)

ood_model = ood_block(cls_model.embed_dim)
ood_dict_pretrained = jt.load('beit_ood_model_B.pkl')
cls_model.load_parameters(ood_dict_pretrained)

# testing dataset loading
split = 'TestSet' + args.split
imgs_dir = '../Dataset/' + split
test_imgs = os.listdir(imgs_dir)

# testing data processing
print('Testing data processing:')

# ipdb.set_trace()
predictions = []
distances = []
with jt.no_grad():
    for img in tqdm(test_imgs):
        img_path = os.path.join(imgs_dir, img)
        image = Image.open(img_path)
        image = transform(image).unsqueeze(0)
        feats, feats25, feats26 = cls_model.forward_features(image)
        prediction = cls_model.head(feats26)
        max_index = jt.argmax(prediction, dim=1)[0]
        norm_l2 = mse_criterion(feats, classes_mean[max_index])
        norm_l225 = mse_criterion(feats25, classes_mean25[max_index])
        norm_l226 = mse_criterion(feats26, classes_mean26[max_index])
        
        distances.append((norm_l2+norm_l225+norm_l226).item())
        if norm_l2 > 0.99:
            image_features = clipmodel.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
            predictions.append(text_probs)
        else:
            sss1 = jt.zeros((1, 28)) - 1000
            predictions.append(jt.cat([prediction, sss1], dim=1))
              
predictions = jt.cat(predictions)
distances = jt.array(distances)
# testing
with open('result.txt', 'w') as save_file:
    i = 0
    for prediction in predictions.tolist():
        prediction = np.asarray(prediction)
        top5_idx = prediction.argsort()[-1:-6:-1]
        save_file.write(test_imgs[i] + ' ' + ' '.join(str(idx) for idx in top5_idx) + '\n')
        i += 1


