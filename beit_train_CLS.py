import jittor as jt
import ipdb
import numpy as np
from jittor import nn
from collections import OrderedDict
from beit_finetune import VisionTransformer
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
    def __init__(self, imgs_dir, data_dir, labels):
        super().__init__()
        self.data_dir = data_dir
        self.labels = labels
        self.imgs_dir = imgs_dir
        self.transform2 = Compose([
            Resize(224, mode=Image.BICUBIC),
            CenterCrop(224), _convert_image_to_rgb,
            RandomHorizontalFlip(),
            RandomErasing(),
            ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                           (0.26862954, 0.26130258, 0.27577711)),
            ImageToTensor()
        ])
        self.transform1 = Compose([
            Resize(224, mode=Image.BICUBIC),
            CenterCrop(224), _convert_image_to_rgb,
            ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                           (0.26862954, 0.26130258, 0.27577711)),
            ImageToTensor()
        ])

    def __getitem__(self, index):
        img = self.data_dir[index]

        img = os.path.join(self.imgs_dir, img)
        image = Image.open(img)
       # 应用基本的预处理
        image1 = self.transform2(image)
        # image2 = self.transform2(image)
        label = self.labels[index]

        return image1, label

    def __len__(self):
        return len(self.labels)


def main_cls():

    transform = transform_(224)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='A')
    
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

    train_dataset = CustomDataset(imgs_dir, new_train_imgs, new_train_labels)
    
    # 创建数据加载器实例
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)    
    
    state_dict_pretrained = jt.load('beitv2_large_patch16_224_pt1k.pkl')
    # 转换预训练参数的数据类型为模型期望的数据类型
    for name, param in state_dict_pretrained.items():
        if param.dtype == 'float16':
            state_dict_pretrained[name] = param.astype('float32')
    
    cls_model = VisionTransformer(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, init_values=0.1, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
    cls_model.load_parameters(state_dict_pretrained)
    cls_model.reset_classifier(375)
    
    total_params = sum(p.numel() for p in cls_model.parameters())
    print("参数量：total_params, ", total_params)
    
    # for name, param in cls_model.named_parameters():
    #     param.requires_grad = True
    
    for name, param in cls_model.named_parameters():
        if "head" in name or "block25" in name or "block26" in name or "stn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # cls_model = nn.DataParallel(cls_model)
    initial_lr = 0.0001
    optimizer = jt.optim.Adam(cls_model.parameters(), lr=initial_lr, weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    
    decay_rate = 0.1
    decay_epoch = 5
    num_epochs = 11
    cls_model.train()
    
    for epoch in range(num_epochs):
        ce_loss = 0
        mse_loss = 0
        cnt = 0
        adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate, decay_epoch)
        for batch_idx, (x, target) in enumerate(train_dataloader):
            # print(batch_idx)
            optimizer.zero_grad()
            target = target.long()   
            out_all = cls_model(x)
            target_all = jt.cat([target,target])
            # print(target_all)
            loss1 = criterion(out_all, target_all)
            
            loss = loss1
    
            optimizer.backward(loss)  # 添加这行以计算梯度
            optimizer.step()  # 使用计算得到的梯度更新模型参数
            ce_loss += loss.item()
            # mse_loss += loss1.item()
            cnt+=1
        epoch_ce_loss = ce_loss / cnt  
        epoch_mse_loss = mse_loss / cnt
        print('Train Epoch: {:03d},  CE_loss: {:.4f}, Loss_MSE: {:.4f}'.format(epoch, epoch_ce_loss, 0))
    
    
    for param in cls_model.parameters():
        param.requires_grad = True
    
    # cls_model = nn.DataParallel(cls_model)
    initial_lr = 0.00001
    optimizer = jt.optim.Adam(cls_model.parameters(), lr=initial_lr, weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()
    
    decay_rate = 0.1
    decay_epoch = 10
    num_epochs = 21
    cls_model.train()
    
    for epoch in range(num_epochs):
        ce_loss = 0
        mse_loss = 0
        cnt = 0
        adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate, decay_epoch)
        for batch_idx, (x, target) in enumerate(train_dataloader):
            # print(batch_idx)
            optimizer.zero_grad()
            target = target.long()                
            out_all = cls_model(x)
            target_all = jt.cat([target,target])
            loss1 = criterion(out_all, target_all)
            
            loss = loss1
    
            optimizer.backward(loss)  # 添加这行以计算梯度
            optimizer.step()  # 使用计算得到的梯度更新模型参数
            ce_loss += loss.item()
            # mse_loss += loss1.item()
            cnt+=1
        epoch_ce_loss = ce_loss / cnt  
        # epoch_mse_loss = mse_loss / cnt
        print('Train Epoch: {:03d},  CE_loss: {:.4f}, Loss_MSE: {:.4f}'.format(epoch, epoch_ce_loss, 0))

    return cls_model

cls_model = main_cls()
jt.save(cls_model.state_dict(), 'beit_cls_model_B.pkl')