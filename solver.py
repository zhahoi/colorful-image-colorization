import os
import time
import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataloader import data_loader, validate_loader
from models.model_ import ColorNet
from models.layers import NNEncLayer, PriorBoostLayer, NonGrayMaskLayer, decode
from loss.focalloss import balanced_focal_loss

from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import os

import random
import torch.backends.cudnn as cudnn

## fix seed
seed = 42
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed+1)
torch.manual_seed(seed+2)
torch.cuda.manual_seed_all(seed+3)

class Solver():
    def __init__(self, root='dataset/anime_faces', result_dir='result', weight_dir='weight', load_weight=False,
                 batch_size=10, test_size=10, img_size=256, num_epoch=100, save_every=1000, save_epoch=2,
                 lr=0.0001, beta_1=0.9, beta_2=0.99, lambda_recon=0.001, logdir=None, lr_decay_iters=100, 
                 pretrained_model='checkpoint_10_epoch.pkl',
                 ):

        # load training dataset
        train_loader, _ = data_loader(root=root, batch_size=batch_size, shuffle=True, 
                                            img_size=img_size, mode='train')
        # load validating dataset
        valid_loader, _ = validate_loader(root=root, batch_size=test_size, shuffle=True, 
                                            img_size=img_size, mode='test')

        # load model
        model = nn.DataParallel(ColorNet()).cuda()
        encode_layer = NNEncLayer()
        boost_layer = PriorBoostLayer()
        nongray_mask = NonGrayMaskLayer()
        
        # loss and optimizer
        # ce_loss = nn.CrossEntropyLoss(reduce=False).cuda()
        # focal_loss = balanced_focal_loss().cuda()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta_1, beta_2), weight_decay=0.001)

        # whether to load weight
        if load_weight is True:
            checkpoint = torch.load(os.path.join(weight_dir, pretrained_model))
            start_epoch = self.load_checkpoint(model, optimizer, checkpoint)

        net_scheduler = self.get_scheduler(optimizer, lr_decay_iters)

        # summarywriter
        writer = SummaryWriter(logdir)

        # create file_path if not exist
        if os.path.exists(result_dir) is False:
            os.makedirs(result_dir)
        if os.path.exists(weight_dir) is False:
            os.makedirs(weight_dir)
        
        # set start epoch
        if load_weight is False:
            start_epoch = 0

        # training start!
        print('====================     Training    Start... =====================')
        for epoch in range(start_epoch, num_epoch):
            start_time = time.time()
            model.train()

            for iter, (images, img_ab) in tqdm(enumerate(train_loader)):
                images = images.unsqueeze(1).float().cuda()
                img_ab = img_ab.float()  # [bs, 2, 56, 56]

                # set model gradients to zero.
                optimizer.zero_grad()

                # preprocess data
                encode, max_encode = encode_layer.forward(img_ab)  # Paper Eq(2) Z空间ground-truth的计算
                targets = torch.Tensor(max_encode).long().cuda()
                boost = torch.Tensor(boost_layer.forward(encode)).float().cuda()  # Paper Eq(3)-(4), [bs, 1, 56, 56], 每个空间位置的ab概率
                mask = torch.Tensor(nongray_mask.forward(img_ab)).float().cuda()  # ab通道数值和小于5的空间位置不计算loss, [bs, 1, 1, 1]
                boost_nongray = boost * mask
                
                predict_q = model(images)

                # compute classification loss
                # loss = (ce_loss(predict_q, targets) * (boost_nongray.squeeze(1))).mean()
                loss = (balanced_focal_loss(logits=predict_q, labels=targets).cuda() * (boost_nongray.squeeze(1))).mean()

                loss.backward()
                optimizer.step()

                # sum iter
                iters = iter + epoch * len(train_loader) + 1

                # ============save the epoch number====================
                log_file = open('log.txt', 'w')
                log_file.write(str(epoch))

                # print error, save intermediate result image and weight
                if iters % save_every == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    print('[Elapsed : %s /Epoch : %d / Iters : %d] => loss : %f'\
                          %(et, epoch, iters, loss.item()))
                    
                    # use validation 
                    model.eval()
                    with torch.no_grad():
                        # get validation data path
                        for iter, (img_original, img_gray, image) in enumerate(valid_loader):
                            """ img_input
                            torch.Size([20, 224, 224, 3]) print(img_original.shape)
                            torch.Size([20, 224, 224, 1]) print(img_gray.shape)
                            torch.Size([20, 1, 224, 224]) print(image.shape)
                            """
                            img_gray = img_gray[iter] / 100 * 255
                            img_gray = img_gray.squeeze(dim=0).numpy()
                            img_gray = np.tile(img_gray, [1,1,3]).astype(np.uint8)
    
                            image = image[iter].float().unsqueeze(dim=0).cuda()
                            img_original = img_original[iter].squeeze(dim=0).numpy().astype(np.uint8)
                            
                            # get predicted q
                            predict_q = model(image)
        
                            ## decode q to original image space
                            color_img = decode(image, predict_q)
                            # print(color_img.shape) (224, 224, 3)
                            color_img = (color_img * 255.).astype(np.uint8)

                            ## save the plot image
                            img_name = '{epoch}_{iters}.png'.format(epoch=epoch, iters=iter)
                            img_path = os.path.join(result_dir, img_name)

                            # predict image
                            plt.figure(figsize=(10,3))
                            
                            plt.subplot(1,3,1)
                            plt.title('Grayscale Image')
                            plt.imshow(img_gray, cmap ='gray')
                            plt.axis('off')
                            plt.xticks([])
                            plt.yticks([])
                            
                            plt.subplot(1,3,2)
                            plt.title('Generated Image')
                            plt.imshow(color_img)
                            plt.axis('off')
                            plt.xticks([])
                            plt.yticks([])
                            
                            plt.subplot(1,3,3)
                            plt.title('Ground Truth')
                            plt.imshow(img_original)
                            plt.axis('off')
                            plt.xticks([])
                            plt.yticks([])
                            
                            plt.tight_layout()
                            # plt.show()
                            plt.savefig(img_path, dpi=600)
                            
                            plt.close()
                    
            # ============plot the loss==================== 
            writer.add_scalars('losses', {'loss': loss}, epoch)

            # ============update the learning rate====================
            self.update_learning_rate(net_scheduler, optimizer)

            # Save weight at the end of every epoch
            if epoch % save_epoch == 0:
                checkpoint = {
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              "epoch": epoch
                              }
                path_checkpoint = os.path.join(weight_dir, "checkpoint_{}_epoch.pkl".format(epoch))
                self.save_checkpoint(checkpoint, path_checkpoint)

        writer.close()

    '''
        < load_checkpoint >
        Load checkpoint
    '''
    def load_checkpoint(self, model, optimizer, checkpoint):
        print('Load pretrained model...')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        return start_epoch

    '''
        < save_checkpoint >
        Save checkpoint
    '''
    def save_checkpoint(self, state, file_name):
        print('saving check_point...')
        torch.save(state, file_name)


    def get_scheduler(self, optimizer, lr_decay_iters):
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.5)
        return scheduler

    # update learning rate (called once every epoch)
    def update_learning_rate(self, scheduler, optimizer):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
