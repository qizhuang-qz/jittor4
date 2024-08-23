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
from beit_train_class_mean import main_protos
from beit_train_CLS import main_cls
# import torch
random.seed(1)
# jt.Module.mpi_param_broadcast(root=0)


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

class PatchLevelAugmentation:
    def __init__(self, patch_size=16, augment_ratio=0.2):
        self.patch_size = patch_size
        self.augment_ratio = augment_ratio

    def random_flip(self, patch):
        if random.choice([True, False]):
            patch = ImageOps.mirror(patch)  # 水平翻转
        if random.choice([True, False]):
            patch = ImageOps.flip(patch)  # 垂直翻转
        return patch

    def random_rotate(self, patch):
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            patch = patch.rotate(angle)
        return patch

    def random_color_jitter(self, patch):
        enhancers = [
            (ImageEnhance.Color, np.random.uniform(0.9, 1.1)),
            (ImageEnhance.Brightness, np.random.uniform(0.9, 1.1)),
            (ImageEnhance.Contrast, np.random.uniform(0.9, 1.1)),
            (ImageEnhance.Sharpness, np.random.uniform(0.9, 1.1)),
        ]
        for enhancer, factor in enhancers:
            patch = enhancer(patch).enhance(factor)
        return patch

    def random_scale(self, patch):
        scale = np.random.uniform(0.8, 1.2)
        w, h = patch.size
        patch = patch.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        patch = patch.resize((w, h), Image.BILINEAR)  # resize back to original size
        return patch

    def augment_patch(self, patch):
        patch = self.random_flip(patch)
        patch = self.random_rotate(patch)
        patch = self.random_color_jitter(patch)
        patch = self.random_scale(patch)
        return patch

    def __call__(self, img):
        img = img.convert('RGB')  # 确保图像是RGB格式
        w, h = img.size
        augmented_img = img.copy()
        patch_positions = [(i, j) for i in range(0, h, self.patch_size) for j in range(0, w, self.patch_size)]
        num_patches_to_augment = int(len(patch_positions) * self.augment_ratio)
        patches_to_augment = random.sample(patch_positions, num_patches_to_augment)

        for i, j in patch_positions:
            box = (j, i, j + self.patch_size, i + self.patch_size)
            patch = img.crop(box)
            if (i, j) in patches_to_augment:
                patch = self.augment_patch(patch)
            augmented_img.paste(patch, box)
        return augmented_img

# 定义函数来随机选择并应用两个增强方法
def apply_random_transforms(image):
    selected_transforms = random.sample(all_transformations, 2)
    for t in selected_transforms:
        image = t(image)
    return image


class CustomDataset(Dataset):
    def __init__(self, data, data25, data26, labels):
        super().__init__()
        self.data = data
        self.data25 = data25
        self.data26 = data26
        self.labels = labels

    def __getitem__(self, index):
        img = self.data[index]
        img25 = self.data25[index]
        img26 = self.data26[index]
        label = self.labels[index]

        return img, img25, img26, label

    def __len__(self):
        return len(self.labels)

def main_ood():
    state_dict_pretrained = jt.load('beit_cls_model_B.pkl')
    cls_model = VisionTransformer(
                patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, init_values=0.1, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
    cls_model.load_parameters(state_dict_pretrained)
    cls_model.reset_classifier(375)
    classes_mean, classes_mean25, classes_mean26 = main_protos(cls_model)
    transform = transform_(224)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='B')
    
    args = parser.parse_args()
    
    # # training data loading
    imgs_dir = '../Dataset/'
    train_labels = open('../Dataset/train.txt').read().splitlines()
    train_imgs = [l.split(' ')[0] for l in train_labels]
    # ipdb.set_trace()
    # 使用列表解析为每个路径分配类别值
    train_labels = jt.array([float(l.split(' ')[1]) for l in train_labels])
    
    data_ids = np.load('closest_ids.npy')
    
    new_train_imgs = np.array(train_imgs)[data_ids]
    new_train_labels = jt.array(train_labels)[data_ids]
    
    ood_model = ood_block(cls_model.embed_dim)
    total_params = sum(p.numel() for p in ood_model.parameters())
    print("参数量：total_params, ", total_params)
    
    train_features = []
    train_features25 = []
    train_features26 = []
    train_labels = []
    print('Training data processing:')
    with jt.no_grad():
        for img, label in zip(new_train_imgs, new_train_labels):
            img = os.path.join(imgs_dir, img)
            image = Image.open(img)
            image = transform(image).unsqueeze(0)            
            image_features, image_features25, image_features26 = cls_model.forward_features(image)
            train_features.append(image_features)
            train_features25.append(image_features25)
            train_features26.append(image_features26)
            train_labels.append(label)
    
    train_features = jt.cat(train_features)
    train_features25 = jt.cat(train_features25)
    train_features26 = jt.cat(train_features26)
    train_labels = jt.array(train_labels)
    
    
    # train_dataset = CustomDataset(imgs_dir, new_train_imgs, new_train_labels)
    train_dataset = CustomDataset(train_features, train_features25, train_features26, train_labels)
    # 创建数据加载器实例
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
    
    # ipdb.set_trace()
    
    mse_criterion = jt.nn.MSELoss()
    
    initial_lr = 0.001
    optimizer_ood = jt.optim.Adam(ood_model.svdd.parameters(), lr=initial_lr, weight_decay=0.00001)
    decay_rate = 0.1
    decay_epoch = 50
    num_epochs = 101
    
    for epoch in range(num_epochs):
        ce_loss = 0
        mse_loss = 0
        cnt = 0
        adjust_learning_rate(optimizer_ood, epoch, initial_lr, decay_rate, decay_epoch)
        for batch_idx, (x, x25, x26, target) in enumerate(train_dataloader):
            optimizer_ood.zero_grad()
            target = target.long()                
            # _, feats, _ = ood_model.svdd(x)
            feats, feats25, feats26 = ood_model(x, x25, x26)
            mean = classes_mean[target]
            mean25 = classes_mean25[target]
            mean26 = classes_mean26[target]
            loss1 = mse_criterion(feats, mean)
            loss2 = mse_criterion(feats25, mean25)
            loss3 = mse_criterion(feats26, mean26)
            loss = loss1+loss2+loss3
    
            optimizer_ood.backward(loss)  # 添加这行以计算梯度
            optimizer_ood.step()  # 使用计算得到的梯度更新模型参数
            mse_loss += loss2.item()
            # mse_loss += loss1.item()
            cnt+=1
        epoch_ce_loss = ce_loss / cnt  
        epoch_mse_loss = mse_loss / cnt
        print('Train Epoch: {:03d},  CE_loss: {:.4f}, Loss_MSE: {:.4f}'.format(epoch, epoch_ce_loss, epoch_mse_loss))

    return ood_model

ood_model = main_cls()
jt.save(ood_model.state_dict(), 'beit_ood_model_B.pkl')

