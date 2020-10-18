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


def load_data(batch_size=2):
    '''
    Load data from file, split, then create data loader
    :param: batch_size
    :return: train_loader, test_loader
    '''
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size

    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # push the randomized training data into the dataloader
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    return train_loader, test_loader


def data_preprocess():
    pass


def solo_train(resnet50_fpn, solo_head, train_loader, optimizer,epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    solo_head.train()

    running_loss = 0
    all_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        # get the inputs
        imgs, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]

        # go through backbone
        backouts = resnet50_fpn(imgs)
        fpn_feat_list = list(backouts.values())
        # make the target

        ## forward
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        ## calculate loss
        loss = solo_head.loss(cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        all_loss += loss.item()
        # pdb.set_trace()

        # print process
        if i % 1 == 0:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1))
            running_loss = 0.0

            exit()

    loss = all_loss / i

    return loss

def solo_eval():
    pass


def main():
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
    learning_rate = 1e-2 / 8  # for batch size 2
    import torch.optim as optim
    optimizer = optim.SGD(solo_head.parameters(),
                          lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # if resume == True:
    #   checkpoint = torch.load(path)
    #   yolo_net.load_state_dict(checkpoint['model_state_dict'])
    #   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #   epoch = checkpoint['epoch']

    # Train your network here
    num_epochs = 1
    # loss_list = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        loss = solo_train(resnet50_fpn, solo_head, train_loader, optimizer,epoch)

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


if __name__ == '__main__':
    main()
