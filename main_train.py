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
import os
import time


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

def solo_train(resnet50_fpn, solo_head, train_loader, optimizer,epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    solo_head.train()

    running_total_loss = 0
    all_total_loss = 0

    running_focal_loss = 0
    all_focal_loss = 0

    running_dice_loss = 0
    all_dice_loss = 0

    for i, data in enumerate(train_loader):
        # print('batch: ', i)
        optimizer.zero_grad()

        # get the inputs
        imgs, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]

        # go through backbone
        with torch.no_grad():
          backouts = resnet50_fpn(imgs.to(device))
          fpn_feat_list = list(backouts.values())
        # print(fpn_feat_list)
        # print("--- %s sec ---" % ((time.time() - start_time)))
        # make the target

        ## forward
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        # print("--- %s sec ---" % ((time.time() - start_time)))

        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        # print("--- %s sec ---" % ((time.time() - start_time)))
        # print(ins_gts_list[0], ins_ind_gts_list[0], cate_gts_list[0])
        ## calculate loss
        total_loss, focal_loss, dice_loss = solo_head.loss(cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list)
        # print("--- %s sec ---" % ((time.time() - start_time)))

        total_loss.backward()
        # print("--- %s sec ---" % ((time.time() - start_time)))

        optimizer.step()
        # print("--- %s sec ---" % ((time.time() - start_time)))

        # print statistics
        running_total_loss += total_loss.item()
        all_total_loss += total_loss.item()

        running_focal_loss += focal_loss.item()
        all_focal_loss += focal_loss.item()

        running_dice_loss += dice_loss.item()
        all_dice_loss += dice_loss.item()

        # print('memory used before delete',torch.cuda.memory_allocated())
        # print('memory reserved before delete',torch.cuda.memory_reserved())
        del total_loss ,focal_loss , dice_loss
        del imgs
        del backouts, fpn_feat_list
        del cate_pred_list, ins_pred_list
        del ins_gts_list, ins_ind_gts_list, cate_gts_list
        torch.cuda.empty_cache()
        # print('memory used after delete',torch.cuda.memory_allocated())
        # print('memory reserved after delete',torch.cuda.memory_reserved())
        # print process
        log_iter = 100
        if i % log_iter == (log_iter-1):  # print every 100 mini-batches
            print('[%d, %5d] total_loss: %.5f focal_loss: %.5f  dice_loss: %.5f' %
                  (epoch + 1, i + 1,
                  running_total_loss / log_iter,
                  running_focal_loss / log_iter,
                  running_dice_loss / log_iter))

            running_total_loss = 0.0
            running_focal_loss = 0.0
            running_dice_loss = 0.0
            print("--- %s minutes ---" % ((time.time() - start_time)/60))
            start_time = time.time()



    total_loss = all_total_loss / i
    focal_loss = all_focal_loss / i
    mask_loss = all_dice_loss / i

    return total_loss, focal_loss, mask_loss

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
    solo_head = SOLOHead(num_classes=4, device=device).to(
        device)  ## class number is 4, because consider the background as one category.

    ## initialize optimizer
    learning_rate = 1e-2  # for batch size 2
    import torch.optim as optim
    optimizer = optim.SGD(solo_head.parameters(),
                          lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    resume = False
    epoch_loaded = 0
    if resume == True:
      path = os.path.join('','solo_epoch_'+str(0))
      checkpoint = torch.load(path)
      solo_head.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      epoch_loaded = checkpoint['epoch']
      print('loaded model')

    # Train your network here
    epoch_loaded += 1
    num_epochs = 1
    total_loss_list = []
    focal_loss_list = []
    mask_loss_list = []
    print('.... start training ....')
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_loss, focal_loss, mask_loss = solo_train(resnet50_fpn, solo_head, train_loader, optimizer,epoch+epoch_loaded)

        total_loss_list.append(total_loss)
        focal_loss_list.append(focal_loss)
        mask_loss_list.append(mask_loss)

        print('**** epoch loss ****', total_loss, focal_loss, mask_loss)
        # test(net, testloader)

        ## save model ###
        path = os.path.join('', 'solo_epoch_' + str(epoch+epoch_loaded))
        torch.save({
            'epoch': epoch + epoch_loaded,
            'model_state_dict': solo_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_loss': total_loss_list,
            'focal_loss': focal_loss_list,
            'mask_loss': mask_loss_list
        }, path)

    print('Finished Training')

    print('total_loss_list',total_loss_list)
    print('focal_loss_list',focal_loss_list)
    print('mask_loss_list',mask_loss_list)


if __name__ == '__main__':
    # start_time = time.time()
    main()
    # print("--- %s minutes ---" % ((time.time() - start_time)/60))

