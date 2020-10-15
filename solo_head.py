import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
import pdb

class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                alpha=0.25,
                                weight=1),
                 postprocess_cfg=dict(cate_thresh=0.2,
                                      ins_thresh=0.5,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out) == len(self.strides)
        pass

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
        # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
        # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
        # self.cate_out is 1 out-layer of conv2d
        # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
    def _init_layers(self):
        ## TODO initialize layers: stack intermediate layer and output layer
        # define groupnorm
        num_groups = 32
        # initial the two branch head modulelist

        #defines the conv layer
        self.simple_conv = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.GroupNorm(32,256),
            nn.ReLU()
        )
        #definee the first conv layer of mask branch
        self.ins_conv_1 = nn.Sequential(
            nn.Conv2d(258,256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.GroupNorm(32,256),
            nn.ReLU()
        )

        self.cate_head = nn.ModuleList([self.simple_conv for _ in range(7)])  #define cnn here
        self.ins_head = nn.ModuleList([self.simple_conv if i!=0 else self.ins_conv_1 for i in range(7)])
        # self.ins_head.append(self.ins_conv_1)
        # for i in range(6):
        #     self.ins_head.append(self.simple_conv)

        self.cate_out = nn.Sequential(
            nn.Conv2d(256,self.cate_out_channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.Sigmoid()
        )

        self.ins_out = nn.ModuleList()
        for num_grid in self.seg_num_grids:
            conv_out = nn.Sequential(
                nn.Conv2d(256, num_grid **2,kernel_size=1,bias=True),
                nn.Sigmoid()
            )
            self.ins_out.append(conv_out)
        pass



    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            # print('weight',m.weight)
    # This function initialize weights for head network

    def bias_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.bias)
            # print('bias',m.bias)
    def _init_weights(self):
        ## TODO: initialize the weights

        #test
        self.cate_head.apply(self.weights_init)
        self.ins_head.apply(self.weights_init)
        # print('initialized convolutional layer weight')

        #initialize the bias of the two conv_out layer to 0
        self.cate_out.apply(self.bias_init)
        self.ins_out.apply(self.bias_init)
        # print('initialized bias')
        pass




    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
        # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
        # if eval = False
            # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
    def forward(self,
                fpn_feat_list,
                eval=False):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256,100,136)
        quart_shape = [new_fpn_list[0].shape[-2]*2, new_fpn_list[0].shape[-1]*2]  # stride: 4
        # print('quart_shape',quart_shape)
        # TODO-Done: use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        # for i in new_fpn_list:
        #     print(i.shape)
        cate_pred_list, ins_pred_list =\
            self.MultiApply(self.forward_single_level,
            new_fpn_list, list(range(len(new_fpn_list))), eval=eval,
            upsample_shape=quart_shape)

        assert len(new_fpn_list) == len(self.seg_num_grids)

        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
        # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):
        #downsample the corsest feature map
        fpn_feat_list[0] = F.interpolate(fpn_feat_list[0],scale_factor=0.5,mode='bilinear')
        #upsample the finest feature map
        fpn_feat_list[-1] = F.interpolate(fpn_feat_list[-1],size=(25,34))
        return fpn_feat_list
        pass


    # This function forward a single level of fpn_featmap through the network
    # Input:
        # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
        # if eval==False
            # cate_pred: (bz,C-1,S,S)
            # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred: (bz,S,S,C-1) / after point_NMS
            # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## TODO-Done: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
        cate_pred = fpn_feat
        ins_pred = fpn_feat
        num_grid = self.seg_num_grids[idx]  # current level grid

        # resize input to  (b, 256, S,S)
        cate_pred = F.interpolate(fpn_feat, size=(num_grid, num_grid))

        # print('cate shape', cate_pred.shape)
        # print(ins_pred.shape)
        # exit()
        # forward process for cate branch
        for cate_conv_layer in self.cate_head:
            cate_pred = cate_conv_layer(cate_pred)
        cate_pred = self.cate_out(cate_pred) # output prediction: (b,(C-1), S, S)

        # concatenate 2 normalized meshgrid x,y
        w = fpn_feat.size()[2]
        h = fpn_feat.size()[3]


        #normalize x,y to (-1,1)
        tensor_w = ((torch.arange(0, w) / float(w)) - 0.5) * 2
        tensor_h = ((torch.arange(0, h) / float(h)) - 0.5) * 2

        mesh_w, mesh_h = torch.meshgrid(tensor_w, tensor_h)
        # should the first index be batch size?
        # mesh_layers = torch.ones([2,2,w,h])
        mesh_layers = torch.ones([cate_pred.shape[0], 2, w, h])

        mesh_layers[:, 0, :, :] = mesh_layers[:, 0, :, :] * mesh_w
        mesh_layers[:, 1, :, :] = mesh_layers[:, 1, :, :] * mesh_h

        ins_pred = torch.cat((ins_pred, mesh_layers),1) # shape (2, 258, w. h)


        # forward process for ins branch
        for ins_conv_layer in self.ins_head:
            ins_pred = ins_conv_layer(ins_pred)
        ins_pred = self.ins_out[idx](ins_pred) # output size: (bz, S^2, H_feat, W_feat)

        # sample output to (bz, S^2, 2H_feat, 2W_feat)
        ins_pred = F.interpolate(ins_pred, scale_factor=2) # expected shape [bz,S^2, 2w, 2h]

        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            ## TODO - Done resize ins_pred
            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1)
            ins_pred = F.interpolate(ins_pred, size=upsample_shape)

        # check flag
        if eval == False:
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
        # heat: (bz,C-1, S, S)
    # Output:
        # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
        # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
        # cate_loss, mask_loss, total_loss
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        ## TODO: compute loss, vecterize this part will help a lot. To avoid potential ill-conditioning, if necessary, add a very small number to denominator for focalloss and diceloss computation.

        ## uniform the expression for ins_gts & ins_preds
        #following code are provided in the pdf instruction
        # for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_gts_list), zip(*ins_ind_gts_list)):
        #     for ins_labels_level_img, ins_ind_labels_level_img in zip(ins_labels_level, ins_ind_labels_level):
        #         print(ins_labels_level_img.shape)
        #         print(ins_ind_labels_level_img.shape)
        #         print(ins_ind_labels_level_img.dtype)
        #         print(ins_labels_level_img[ins_ind_labels_level_img.type(torch.long)].shape)
        #         exit()
        ins_gts = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img.type(torch.long), ...]
                              for ins_labels_level_img, ins_ind_labels_level_img in
                              zip(ins_labels_level, ins_ind_labels_level)], 0)
                   for ins_labels_level, ins_ind_labels_level in
                   zip(zip(*ins_gts_list), zip(*ins_ind_gts_list))]
        ## ins_gts: fpn x batch_size * s^2 x 2H_feat x 2W_feat
        ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img.type(torch.long), ...]
                                for ins_preds_level_img, ins_ind_labels_level_img in
                                zip(ins_preds_level, ins_ind_labels_level)], 0)
                     for ins_preds_level, ins_ind_labels_level in
                     zip(ins_pred_list, zip(*ins_ind_gts_list))]
        ## ins_gts: fpn x batch_size * s^2 x 2H_feat x 2W_feat


        ## uniform the expression for cate_gts & cate_preds
        # cate_gts: (bz*fpn*S^2,), img, fpn, grids
        # cate_preds: (bz*fpn*S^2, C-1), ([img, fpn, grids], C-1)
        cate_gts = [torch.cat([cate_gts_level_img.flatten()
                               for cate_gts_level_img in cate_gts_level])
                    for cate_gts_level in zip(*cate_gts_list)]
        cate_gts = torch.cat(cate_gts)
        cate_preds = [cate_pred_level.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
                      for cate_pred_level in cate_pred_list]
        cate_preds = torch.cat(cate_preds, 0)

        #cate_gts: (N,)
        #cate_preds: (N,3)

        cate_loss = self.FocalLoss(cate_preds,cate_gts)
        num_fpn = len(ins_gts)

        dice_loss_sum = 0
        N_total = 0
        #assuming activated grid cells have been picked out using TA's code, could be wrong
        for i in range(num_fpn):
            dice_loss_sum += self.DiceLoss(ins_preds[i],ins_gts[i])
            N_total += ins_preds[i].shape[0]
        mask_loss = dice_loss_sum / N_total
        pass



    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # changed the input to fpn x batch_size * s^2 x 2H_feat x 2W_feat for vectorization
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        # N_total = mask_pred.shape[0]
        # for i in range(len(mask_pred)):
        #     N_total += mask_pred[i].shape[0]
        #     print('mask_pred',mask_pred[i].shape)
        #     print('mask_gt',mask_gt[i].shape)
        # print('N_total',N_total)
        dice_loss = 2 * torch.sum(mask_pred * mask_gt,dim=(-1,-2)) / (torch.sum(mask_pred ** 2,dim=(-1,-2)) + torch.sum(mask_gt ** 2,dim=(-1,-2)) + 1e-5)  #added 1e-5 to prevent devide by 0
        dice_loss = 1 - dice_loss
        dice_loss = torch.sum(dice_loss)
        return dice_loss


    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        ## TODO: compute focalloss
        expanded_cate_gts = torch.zeros((len(cate_gts),4))
        expanded_cate_gts[:,cate_gts.type(torch.long)] = 1 # one hot encoding
        flat_cate_gts = expanded_cate_gts[:,1:].flatten() #remove the background channel and flatten
        flat_cate_preds = cate_preds.flatten()
        total_num = len(flat_cate_preds)
        alpha = self.cate_loss_cfg.get('alpha')
        gamma = self.cate_loss_cfg.get('gamma')
        fl_1 = -alpha * (1-flat_cate_gts[torch.where(flat_cate_gts == 1)] ** gamma ) * torch.log(flat_cate_gts[torch.where(flat_cate_gts == 1)]) # for y_i = 1
        fl_2 = -(1-alpha) * (flat_cate_gts[torch.where(flat_cate_gts != 1)] ** gamma) * torch.log(1-flat_cate_gts[torch.where(flat_cate_gts != 1)]) # for y_i != 1
        focal_loss = (torch.sum(fl_1) + torch.sum(fl_2))/total_num

        return focal_loss
        pass

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # / ins_pred_list is only used to record feature map
        # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
        # label_list: list, len(batch_size), each (n_object, )
        # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,
               ins_pred_list,
               bbox_list,
               label_list,
               mask_list):
        # TODO: use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list. Parallel w.r.t. img mini-batch

        # remember, you want to construct target of the same resolution as prediction output in training

        featmap_sizes = [[(featmap.shape[2],featmap.shape[3]) for featmap in ins_pred_list ]] * len(mask_list)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(self.targer_single_img, \
                                                                              bbox_list, label_list,
                                                                              mask_list, featmap_sizes
                                                                              )

        # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list
    # -----------------------------------
    ## process single image in one batch
    # -----------------------------------
    # input:
        # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
        # gt_labels_raw: n_obj,
        # gt_masks_raw: n_obj, H_ori, W_ori
        # featmap_sizes: list of shapes of featmap
    # output:
        # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
        # cate_label_list: list, len: len(FPN), (S, S)
        # ins_ind_label_list: list, len: len(FPN), (S^2, )
    def targer_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None):
        ## TODO: finish single image target build
        # compute the area of every object in this single image

        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []

        # initial the output list, each entry for one featmap

        w = torch.abs(gt_bboxes_raw[:,0] - gt_bboxes_raw[:,2])
        h = torch.abs(gt_bboxes_raw[:,3] - gt_bboxes_raw[:,1])
        scale = torch.sqrt(w * h)
        center_list = np.asarray([ndimage.measurements.center_of_mass(mask.numpy()) for mask in gt_masks_raw])

        #go through each layer of feature pyramid
        for i in range(len(self.seg_num_grids)):
            cate_label = torch.zeros((self.seg_num_grids[i],self.seg_num_grids[i]))
            ins_label = torch.zeros((self.seg_num_grids[i]**2,featmap_sizes[i][0],featmap_sizes[i][1]))
            ins_ind_label = torch.zeros(self.seg_num_grids[i]**2)
            scale_range = self.scale_ranges[i]

            #check object scale
            #changed single expression to double to handle multiple object case
            idx = torch.where((scale_range[0] < scale)  &( scale< scale_range[1]))[0].tolist()


            if len(idx)==0:
                cate_label_list.append(cate_label)
                ins_label_list.append(ins_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            labels = gt_labels_raw[idx]
            ins_layers = gt_masks_raw[idx]
            center = torch.Tensor(center_list[idx]).view(-1,2)
            center_x = center[:,0]
            center_y = center[:,1]
            w *= 0.2
            h *= 0.2

            x_tl = center_x - w/2 # shape(#obj, 1)
            y_tl = center_y - h/2
            x_br = center_x + w/2
            y_br = center_y + h/2

            x_tl_grid = x_tl//(800//self.seg_num_grids[i])
            y_tl_grid = y_tl//(1088//self.seg_num_grids[i])
            x_br_grid = x_br//(800//self.seg_num_grids[i])
            y_br_grid = y_br//(1088//self.seg_num_grids[i])

            center_x /= 800/self.seg_num_grids[i]
            center_y /= 1088/self.seg_num_grids[i]


            zero_msk = torch.zeros((self.seg_num_grids[i], self.seg_num_grids[i]))

            # print('num of objects in this layer', len(idx))
            # print('object label', labels)
            # print('ins_layers', ins_layers.shape)

            #loop through each object thats falls in the range for cate_label
            for k in range(len(labels)):
                # creat cate_label
                temp_cate_label = torch.zeros_like(cate_label)
                temp_cate_label[int(x_tl_grid[k]):int(x_br_grid[k])+1,
                           int(y_tl_grid[k]):int(y_br_grid[k])+1] = labels[k]
                cate_label[temp_cate_label > 0 ] = labels[k]
                zero_msk[int(center_x[k])-1:int(center_x[k])+2,
                         int(center_y[k])-1:int(center_y[k])+2] = 1

                channel_inds = (temp_cate_label == labels[k]).nonzero()
                channels = channel_inds[:,0] * self.seg_num_grids[i] + channel_inds[:,1]
                ins_layer = ins_layers[k]  #assign corresponding mask
                ins_layer = ins_layer.expand(1,1,ins_layer.shape[0],ins_layer.shape[1])
                ins_layer = F.interpolate(ins_layer, size = featmap_sizes[i],mode='bilinear')
                ins_layer = torch.squeeze(ins_layer)
                ins_layer[ins_layer > 0] = 1
                ins_layer = torch.clamp(ins_layer, 0, 1)
                # check this line why index 0  ?
                ins_label[channels, :, :] = ins_layer
                ins_ind_label[channels] = 1

            # turn off grids outside 3x3
            cate_label *= zero_msk
            cate_label_list.append(cate_label)


            ins_label_list.append(ins_label)
            ins_ind_label_list.append(ins_ind_label)

        # check flag
        assert ins_label_list[1].shape == (1296,200,272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)
        return ins_label_list, ins_ind_label_list, cate_label_list

    # This function receive pred list from forward and post-process
    # Input:
        # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
        # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
        # ori_size: [ori_H, ori_W]
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size):

        ## TODO: finish PostProcess
        pass


    # This function Postprocess on single img
    # Input:
        # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
        # cate_pred_img: (all_level_S^2, C-1)
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size):

        ## TODO: PostProcess on single image.
        pass

    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS
        pass

    # -----------------------------------
    ## The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
        # color_list: list, len(C-1)
        # img: (bz,3,Ori_H, Ori_W)
        ## self.strides: [8,8,16,32,32]
    def PlotGT(self,
               ins_gts_list,
               ins_ind_gts_list,
               cate_gts_list,
               color_list,
               img):
        ## TODO: target image recover, for each image, recover their segmentation in 5 FPN levels.
        ## This is an important visual check flag.

        # print(len(img))
        #loop through imgs in one batch
        for i in range(len(ins_gts_list)):
            #loop through all featurn pyramid layers:
            for j in range(len(ins_gts_list[i])):
                fig, ax = plt.subplots(1)
                image = img[i]
                image = image.permute(1,2,0)
                ax.imshow(image)
                for k in range(3):
                    color = color_list[k]
                    k = k + 1
                    idx = torch.where(cate_gts_list[i][j] == k)

                    if len(idx[0]) == 0:
                        continue
                    flat_idx = idx[0]*self.seg_num_grids[j] + idx[1]
                    msk_list = ins_gts_list[i][j][flat_idx]

                    for msk in msk_list:

                        #4d bilinear interpolation, upsasmpling to original size
                        msk = msk.expand(1,1,msk.shape[0],msk.shape[1])
                        msk = F.interpolate(msk, size=(image.shape[0],image.shape[1]), mode='bilinear')
                        msk = torch.squeeze(msk)
                        #clipping and set msk data to binary as the original format
                        msk[msk > 0] = 1
                        msk = torch.clamp(msk,0,1)
                        msk = np.ma.masked_where(msk == 0, msk)
                        #alpha set to 1.0 for color consistence, multiple grid activated would have solider color
                        ax.imshow(msk, color, interpolation='none',alpha=1.0)


                # fig.savefig("./testfig/gt_visualization_fp"+str(i)+"_"+str(j)+".png" )
                # plt.show()

        pass

    # This function plot the inference segmentation in img
    # Input:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        # color_list: ["jet", "ocean", "Spectral"]
        # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        ## TODO: Plot predictions
        pass

from backbone import *
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
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

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2 #used 8 for part a
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target


        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        print('cate_gts_list',cate_gts_list[0][0].shape)
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)


        #testing the loss function here:
        solo_head.loss(cate_pred_list,ins_pred_list,ins_gts_list,ins_ind_gts_list,cate_gts_list)
        print('tested loss function')


        #compare with raw label result
        # for i in range(batch_size):
        #     image = img[i]
        #     image = image.permute(1, 2, 0)
        #     fig, ax = plt.subplots(1)
        #     ### Display the image ###
        #     ax.imshow(np.squeeze(image))
        #
        #     # plot mask
        #
        #     for j, msk in enumerate(mask_list[i]):
        #         cls = label_list[i][j]-1
        #         msk = np.ma.masked_where(msk == 0, msk)
        #         plt.imshow(msk, mask_color_list[int(cls)], interpolation='none', alpha=0.7)
        #
        #     # plt.savefig("./testfig/visualtrainset"+str(iter)+"_"+ str(i)+".png")
        #     plt.show()

        exit()

