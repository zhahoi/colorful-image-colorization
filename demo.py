#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import argparse
import torch
import os.path as osp
from re import I
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray

import torch.nn as nn
from models.layers import decode
from models.model_ import ColorNet
from torchvision import transforms

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    if transform is not None:
        image = transform(image)
    image = np.asarray(image)
    img_original = copy.deepcopy(image)
    image_gray = rgb2lab(image)[:, :, 0]
    image = image_gray - 50.
    image = torch.Tensor(image).unsqueeze(0)
    
    return img_original, image_gray, image


def main(args):
    ## Data
    data_dir = osp.join(args.data_path, args.split)
    file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            img_path = file
            file_list.append(img_path)
    
    ## Model
    color_model = nn.DataParallel(ColorNet()).cuda()
    color_model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    color_model.eval()
    print("load pdparams successfully")

    transform = transforms.Compose([
                                    transforms.Resize((args.img_size, args.img_size)),
                                   ])

    with torch.no_grad():
        ## Inference
        for index, file in enumerate(file_list):
            ## 读取数据
            img_original, img_gray, image = load_image(data_dir + "/" + file, transform=transform)
            img_gray = img_gray / 100 * 255
            img_gray = img_gray.astype(np.uint8)
            image = image.float().unsqueeze(0).cuda()
            ## 获得ab量化空间预测
            img_ab_313 = color_model(image)
            ## 将量化空间解码为原空间
            color_img = decode(image, img_ab_313)
            color_img = color_img * 255.
            color_img = color_img.astype(np.uint8)
            ## 保存结果
            if not osp.exists(osp.join(args.save_path, args.split)):
                os.mkdir(osp.join(args.save_path, args.split))
            save_name = osp.join(args.save_path, args.split, file)
            plt.figure(figsize=(10, 3))
            
            plt.subplot(1,3,1)
            plt.imshow(img_original)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1,3,2)
            plt.imshow(img_gray, cmap ='gray')
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1,3,3)
            plt.imshow(color_img)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            
            plt.tight_layout()
            plt.savefig(save_name, dpi=600)
            plt.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorization!')
    parser.add_argument('--data_path', type=str, default='C:/Dataset/NCDataset/',
                        help='path to dataset splits data folder')
    parser.add_argument('--split', type=str, default='test', help='dataset split')
    parser.add_argument('--model_path', default='C:/Code/black2color/weights/checkpoint_120_epoch.pkl', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--save_path', type=str, default='results',
                        help='path to save results')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='Image size')
    args = parser.parse_args()
   
    main(args)
