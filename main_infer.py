'''
Main training process SOLO
Author: Chang Liu, Qi Jin
Date: 10/15/2020
Instructions:
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
import pdb
from backbone import *
from solo_head import *
from main_train import *
import os


def load_model(net, optimizer, device,epoch_num=0):
    path = os.path.join('', 'solo_epoch_' + str(epoch_num))
    checkpoint = torch.load(path,map_location=torch.device(device))

    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return net, optimizer

def infer(solo_head,resnet50_fpn, test_loader, device='cpu', mode='infer'):
    solo_head.eval()
    resnet50_fpn.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # get the inputs
            imgs, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]

            # go through backbone
            backouts = resnet50_fpn(imgs.to(device))
            fpn_feat_list = list(backouts.values())

            ## forward
            cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=True)

            NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list = solo_head.PostProcess(
                    ins_pred_list,
                    cate_pred_list,
                    ori_size=imgs.shape[-2:])


            if mode=='infer':
                pass
            if mode=='map':
                pass




def main_infer(mode='infer'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ## load data
    train_loader, test_loader = load_data(batch_size=2)
    print('data loaded')

    ## initialize backbone and SOLO head
    resnet50_fpn = Resnet50Backbone().to(device)
    solo_head = SOLOHead(num_classes=4).to(
        device)  ## class number is 4, because consider the background as one category.

    ## initialize optimizer
    learning_rate = 1e-2  # for batch size 2
    import torch.optim as optim
    optimizer = optim.SGD(solo_head.parameters(),
                          lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    ## only check the final model we have, output example figures
    if mode == 'infer':
        solo_head, optimizer = load_model(solo_head, optimizer, device, epoch_num=0)
        infer(solo_head,resnet50_fpn, test_loader, device=device, mode='infer')



        pass
    ## track map over all epoch of model on test dataset
    if mode == 'map':
        pass


if __name__ == '__main__':
    main_infer()
