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

def load_model(net, optimizer, epoch_num=0):

    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return net


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
        solo_head, optimizer = load_model(solo_head, optimizer, epoch_num=0)

        pass
    ## track map over all epoch of model on test dataset
    if mode == 'map':
        pass
    # if resume == True:
    #   checkpoint = torch.load(path)
    #   yolo_net.load_state_dict(checkpoint['model_state_dict'])
    #   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #   epoch = checkpoint['epoch']

    # Train your network here
    num_epochs = 1
    # loss_list = []
    print('.... start training ...')
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        loss = solo_train(resnet50_fpn, solo_head, train_loader, optimizer, epoch)

        # loss_list.append(loss)

        print('**** epoch loss', loss)
        # test(net, testloader)

        ### save model ###
        # path = os.path.join('', 'yolo_epoch' + str(epoch))
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': net.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict()
        # }, path)

    print('Finished Training')

    pass


if __name__ == '__main__':
    main_infer()
