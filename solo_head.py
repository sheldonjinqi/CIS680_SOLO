import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
# import pdb

class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 device='cpu',
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
        self.device = device

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        # assert len(self.ins_head) == self.stacked_convs
        # assert len(self.cate_head) == self.stacked_convs
        # assert len(self.ins_out) == len(self.strides)
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



    def forward(self,
                fpn_feat_list,
                eval=False):
        '''
          Forward function should forward every levels in the FPN.
          this is done by map function or for loop
          Input:
              fpn_feat_list: backout_list of resnet50-fpn
          Output:
              if eval = False
                  cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
                  ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
              if eval==True
                  cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
                  ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
        '''
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        # assert new_fpn_list[0].shape[1:] == (256,100,136)
        quart_shape = [new_fpn_list[0].shape[-2]*2, new_fpn_list[0].shape[-1]*2]  # stride: 4
        # print('quart_shape',quart_shape)
        # TODO-Done: use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        # for i in new_fpn_list:
        #     print(i.shape)
        cate_pred_list, ins_pred_list =\
            self.MultiApply(self.forward_single_level,
            new_fpn_list, list(range(len(new_fpn_list))), eval=eval,
            upsample_shape=quart_shape)

        # assert len(new_fpn_list) == len(self.seg_num_grids)

        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        # assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        # assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    def NewFPN(self, fpn_feat_list):
        '''
          This function upsample/downsample the fpn level for the network
          In paper author change the original fpn level resolution
          Input:
              fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
          Output:
          new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
        '''

        #downsample the corsest feature map
        fpn_feat_list[0] = F.interpolate(fpn_feat_list[0],scale_factor=0.5,mode='bilinear')
        #upsample the finest feature map
        fpn_feat_list[-1] = F.interpolate(fpn_feat_list[-1],size=(25,34))
        return fpn_feat_list


    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        '''
          This function forward a single level of fpn_featmap through the network
          Input:
              fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
              idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
          Output:
              if eval==False
                  cate_pred: (bz,C-1,S,S)
                  ins_pred: (bz, S^2, 2H_feat, 2W_feat)
              if eval==True
                  cate_pred: (bz,S,S,C-1) / after point_NMS
                  ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
        '''
        # upsample_shape is used in eval mode
        ## TODO-Done: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
        # cate_pred = fpn_feat
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
        tensor_w = (((torch.arange(0, w) / float(w)) - 0.5) * 2).to(self.device)
        tensor_h = (((torch.arange(0, h) / float(h)) - 0.5) * 2).to(self.device)

        mesh_w, mesh_h = torch.meshgrid(tensor_w, tensor_h)
        mesh_w, mesh_h = mesh_w.to(self.device), mesh_h.to(self.device)
        # should the first index be batch size?
        # mesh_layers = torch.ones([2,2,w,h])
        mesh_layers = torch.ones([cate_pred.shape[0], 2, w, h]).to(self.device)
        mesh_layers[:, 0, :, :] = mesh_layers[:, 0, :, :] * mesh_w
        mesh_layers[:, 1, :, :] = mesh_layers[:, 1, :, :] * mesh_h
        ins_pred = torch.cat((ins_pred, mesh_layers),1) # shape (2, 258, w. h)


        # forward process for ins branch, loop over 5 head
        for ins_conv_layer in self.ins_head:
            ins_pred = ins_conv_layer(ins_pred)
        ins_pred = self.ins_out[idx](ins_pred) # output size: (bz, S^2, H_feat, W_feat)

        # sample output to (bz, S^2, 2H_feat, 2W_feat)
        ins_pred = F.interpolate(ins_pred, scale_factor=2) # expected shape [bz,S^2, 2w, 2h]

        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            ## TODO - Done resize ins_pred
            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1) # shape (2,40,40,3)
            ins_pred = F.interpolate(ins_pred, size=upsample_shape) # shape (2,1600,200, 272)

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
        ins_gts = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...]
                              for ins_labels_level_img, ins_ind_labels_level_img in
                              zip(ins_labels_level, ins_ind_labels_level)], 0)
                   for ins_labels_level, ins_ind_labels_level in
                   zip(zip(*ins_gts_list), zip(*ins_ind_gts_list))]
        ## ins_gts: list fpn , num all activated cells in the batch x 2H_feat x 2W_feat
        # for level in range(5):
        #     print('ins_gts',ins_gts[level].shape, ' at level', level+1)

        ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]
                                for ins_preds_level_img, ins_ind_labels_level_img in
                                zip(ins_preds_level, ins_ind_labels_level)], 0)
                     for ins_preds_level, ins_ind_labels_level in
                     zip(ins_pred_list, zip(*ins_ind_gts_list))]
        # for level in range(5):
        #     print('ins_preds',ins_preds[level].shape, ' at level', level+1)
        ## ins_preds: list fpn , num all activated cells in the batch x 2H_feat x 2W_feat


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

        # print('cate_loss, mask_loss:', cate_loss, mask_loss)
        return 3* mask_loss + cate_loss, cate_loss, 3*mask_loss



    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # changed the input to num all activated cells in the batch x 2H_feat x 2W_feat for vectorization
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        # N_total = mask_pred.shape[0]
        # for i in range(len(mask_pred)):
        #     N_total += mask_pred[i].shape[0]
        #     print('mask_pred',mask_pred[i].shape)
        #     print('mask_gt',mask_gt[i].shape)
        # print('N_total',N_total)

        mask_pred, mask_gt = mask_pred.to(self.device), mask_gt.to(self.device)
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
        # print(cate_gts.int())
        expanded_cate_gts[range(len(cate_gts)),cate_gts.long()] = 1 # one hot encoding
        flat_cate_gts = expanded_cate_gts[:,1:].flatten() #remove the background channel and flatten
        flat_cate_preds = cate_preds.flatten()
        total_num = len(flat_cate_preds)
        alpha = self.cate_loss_cfg.get('alpha')
        gamma = self.cate_loss_cfg.get('gamma')
        fl_1 = -alpha * ((1-flat_cate_preds[torch.where(flat_cate_gts == 1)]) ** gamma ) * torch.log(flat_cate_preds[torch.where(flat_cate_gts == 1)]) # for y_i = 1
        fl_2 = -(1-alpha) * (flat_cate_preds[torch.where(flat_cate_gts != 1)] ** gamma) * torch.log(1-flat_cate_preds[torch.where(flat_cate_gts != 1)]) # for y_i != 1

        focal_loss = (torch.sum(fl_1) + torch.sum(fl_2))/total_num
        return focal_loss

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))


    def target(self,
               ins_pred_list,
               bbox_list,
               label_list,
               mask_list):
        '''
        This function build the ground truth tensor for each batch in the training
        Input:
            ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
            / ins_pred_list is only used to record feature map
            bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
            label_list: list, len(batch_size), each (n_object, )
            mask_list: list, len(batch_size), each (n_object, 800, 1088)
        Output:
            ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
            ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
            cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
        '''
        # TODO: use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list. Parallel w.r.t. img mini-batch

        # remember, you want to construct target of the same resolution as prediction output in training

        featmap_sizes = [[(featmap.shape[2],featmap.shape[3]) for featmap in ins_pred_list ]] * len(mask_list)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(self.targer_single_img, \
                                                                              bbox_list, label_list,
                                                                              mask_list, featmap_sizes
                                                                              )

        # check flag
        # assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        # assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        # assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

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

        w = torch.abs(gt_bboxes_raw[:,0] - gt_bboxes_raw[:,2]).to(self.device)
        h = torch.abs(gt_bboxes_raw[:,3] - gt_bboxes_raw[:,1]).to(self.device)
        scale = torch.sqrt(w * h)
        center_list = np.asarray([ndimage.measurements.center_of_mass(mask.numpy()) for mask in gt_masks_raw])

        #go through each layer of feature pyramid -> loop of 5
        for i in range(len(self.seg_num_grids)):
            cate_label = torch.zeros((self.seg_num_grids[i],self.seg_num_grids[i])).to(self.device)
            ins_label = torch.zeros((self.seg_num_grids[i]**2,featmap_sizes[i][0],featmap_sizes[i][1])).to(self.device)
            ins_ind_label = torch.zeros(self.seg_num_grids[i]**2,dtype=torch.bool).to(self.device)
            scale_range = self.scale_ranges[i]

            #check object scale
            #changed single expression to double to handle multiple object case
            idx = torch.where((scale_range[0] < scale)  &( scale< scale_range[1]))[0].tolist()

            if len(idx)==0:
                cate_label_list.append(cate_label)
                ins_label_list.append(ins_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            labels = gt_labels_raw[idx].to(self.device)
            ins_layers = gt_masks_raw[idx].to(self.device)
            center = torch.Tensor(center_list[idx]).view(-1,2).to(self.device)
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


            zero_msk = torch.zeros((self.seg_num_grids[i], self.seg_num_grids[i])).to(self.device)

            # print('num of objects in this layer', len(idx))
            # print('object label', labels)
            # print('ins_layers', ins_layers.shape)

            #loop through each object thats falls in the range for cate_label
            for k in range(len(labels)):
                # creat cate_label
                temp_cate_label = torch.zeros_like(cate_label).to(self.device)
                temp_cate_label[int(x_tl_grid[k]):int(x_br_grid[k])+1,
                           int(y_tl_grid[k]):int(y_br_grid[k])+1] = labels[k]
                cate_label[temp_cate_label > 0] = labels[k]
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
                ins_ind_label[channels] = True

            # turn off grids outside 3x3
            cate_label *= zero_msk
            cate_label_list.append(cate_label)

            ins_label_list.append(ins_label)
            ins_ind_label_list.append(ins_ind_label)

        # check flag
        # assert ins_label_list[1].shape == (1296,200,272)
        # assert ins_ind_label_list[1].shape == (1296,)
        # assert cate_label_list[1].shape == (36, 36)

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
        ## finished, needs to be tested, can be vectorized to process entire batch together once tested PostProcessImg

        # ins_pred_list = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img.type(torch.long), ...]
        #                         for ins_preds_level_img, ins_ind_labels_level_img in
        #                         zip(ins_preds_level, ins_ind_labels_level)], 0)
        #              for ins_preds_level, ins_ind_labels_level in
        #              zip(ins_pred_list, zip(*ins_ind_gts_list))]
        # print('memory used before PostProcess', torch.cuda.memory_allocated())
        # print('memory reserved before PostProcess',torch.cuda.memory_reserved())
        batch_size = ins_pred_list[0].shape[0] # should be 2
        NMS_sorted_scores_list = []
        NMS_sorted_cate_label_list = []
        NMS_sorted_ins_list = []
        ins_preds = torch.cat(ins_pred_list,dim=1) #desired output shape: batch_size, all_level s^2, Ori_H/4, Ori_W/4


        cate_preds = [torch.flatten(cate_pred_level,start_dim=1,end_dim=2)
                      for cate_pred_level in cate_pred_list] #list, len fpn, (bz,s^2, c-1)
        cate_preds = torch.cat(cate_preds, 1) #resulting shape should be  (bz,all_level_s^2,c-1)
        # loop through images in batch
        for i in range(batch_size):
            NMS_sorted_scores, NMS_sorted_cate_label, NMS_sorted_ins = self.PostProcessImg(ins_preds[i],cate_preds[i],ori_size)
            NMS_sorted_scores_list.append(NMS_sorted_scores)
            NMS_sorted_cate_label_list.append(NMS_sorted_cate_label)
            NMS_sorted_ins_list.append(NMS_sorted_ins)

        del ins_preds, cate_preds
        torch.cuda.empty_cache()


        assert len(NMS_sorted_scores_list) == batch_size #check output shape
        return NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list


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

        # ## TODO: PostProcess on single image.
        # Here the function can be modified to process the entire batch together
        ins_pred_img = ins_pred_img.cpu()
        cate_pred_img = cate_pred_img.cpu()
        # print('memory used before PostProcessImg',torch.cuda.memory_allocated())
        # print('memory reserved before PostProcessImg',torch.cuda.memory_reserved())

        # print('memory reserved before delete',torch.cuda.memory_reserved())

        c_max,_ = torch.max(cate_pred_img, dim=1)  # maximum category prediction for each cell

        cate_thresh_idx = torch.where(c_max > self.postprocess_cfg.get('cate_thresh')) #find cells that c_max > cate_thresh
        ins_thresh_idx = torch.where(ins_pred_img > self.postprocess_cfg.get('ins_thresh'))

        indicator_fc = torch.zeros_like(ins_pred_img)
        indicator_cate = torch.zeros_like(ins_pred_img)

        ins_pred_img_tmp = ins_pred_img[cate_thresh_idx]
        cate_pred_img_tmp = cate_pred_img[cate_thresh_idx]

        indicator_fc[ins_thresh_idx] = 1

        indicator_cate[cate_thresh_idx] =1

        print('test')

        #vectorized, check the shape of each term, could be bug here!
        mask_score = torch.sum(ins_pred_img * indicator_fc * indicator_cate, dim =(-1,-2)) / \
                     torch.sum(indicator_fc * indicator_cate, dim=(-1,-2)) * c_max

        # not sure why these should be list of len(bz), asked on piazza, double check !
        sorted_mask_score, sort_idx = torch.sort(mask_score,descending= True)
        sorted_cate_pred = cate_pred_img[sort_idx]
        sorted_ins_pred = ins_pred_img[sort_idx]

        NMS_score = self.MatrixNMS(sorted_ins_pred,sorted_mask_score)
        # print('NMS_score',NMS_score)
        sorted_NMS_score, NMS_idx = torch.sort(NMS_score, descending = True)
        sorted_cate_pred = sorted_cate_pred[NMS_idx]
        sorted_ins_pred = sorted_ins_pred[NMS_idx]
        sorted_NMS_score = sorted_mask_score[NMS_idx]

        #pick the 5 highest
        decay_threshold = 0.2
        keep_instance = max(min((NMS_score > decay_threshold).sum(), 5), 1)
        keep_instance = 5
        sorted_NMS_score = sorted_NMS_score[:keep_instance]
        sorted_cate_pred = sorted_cate_pred[:keep_instance]
        # sorted_ins_pred = sorted_ins_pred[:keep_instance] ##### bug!

        _,sorted_cate_pred = torch.max(sorted_cate_pred, dim=1)

        # nms_sorted_ins = F.interpolate(nms_sorted_ins.view(1, keep_instance, ins_h, ins_w), (ori_h, ori_w)) ##### bug!

        print(sorted_NMS_score, sorted_cate_pred, sorted_ins_pred)
        return sorted_NMS_score, sorted_cate_pred, sorted_ins_pred


        # # ## TODO: PostProcess on single image.
        # # # PostProcessing Parameters
        # cate_thresh = self.postprocess_cfg['cate_thresh']
        # ins_thresh = self.postprocess_cfg['ins_thresh']
        # pre_NMS_num = self.postprocess_cfg['pre_NMS_num']
        # keep_instance = self.postprocess_cfg['keep_instance']
        # IoU_thresh = self.postprocess_cfg['IoU_thresh']
        # decay_thresh = 0.2  # suppress all decay scores smaller than decay_thresh
        #
        # # Calculating scores. NMS here means pre-suppress by threshold
        # cate_value, cate_ind = torch.max(cate_pred_img, dim=1)  # (all_level_S^2,), float, [0,1]; int
        # valid_cate_indices = cate_value > cate_thresh  # (all_level_S^2,), bool
        # cate_value_nms = torch.where(valid_cate_indices, cate_value,
        #                              torch.zeros_like(cate_value))  # (all_level_S^2,), float
        # cate_ind_nms = torch.where(valid_cate_indices, cate_ind,
        #                            3 * torch.ones_like(cate_ind))  # (all_level_S^2,), int
        # valid_ins_indices = ins_pred_img > ins_thresh  # (all_level_S^2, ori_H/4, ori_W/4), bool
        # ins_nms = torch.where(valid_ins_indices, ins_pred_img,
        #                       torch.zeros_like(ins_pred_img))  # (all_level_S^2, ori_H/4, ori_W/4), float
        # scores = ins_nms.sum(dim=[1, 2]) / (valid_ins_indices.sum(dim=[1, 2]) + 1e-5) \
        #          * cate_value_nms  # (all_level_S^2,), float
        # scores_order = torch.argsort(scores, descending=True)  # (all_level_S^2,), int
        # scores_order = scores_order[0:pre_NMS_num]  # (n_act,), int
        #
        # # Sort scores
        # sorted_scores = scores[scores_order]  # (n_act,), float
        # sorted_ins = (ins_nms[scores_order] > 0.5).float()  # (n_act, ori_H/4, ori_W/4), [0,1]
        # sorted_cate = cate_ind_nms[scores_order]  # (n_act,), int
        #
        # # Matrix NMS
        # NMS_sorted_scores_list = []
        # NMS_sorted_cate_label_list = []
        # NMS_sorted_ins_list = []
        #
        # decay_scores = self.MatrixNMS(sorted_ins, sorted_scores)  # (n_act,), float
        # keep_instance = max(min((decay_scores > decay_thresh).sum(), keep_instance),1)
        # print(decay_scores)
        # decay_order = torch.argsort(decay_scores, descending=True)  # (n_act,), int
        # decay_order = decay_order[0:keep_instance]  # (keep_instance,), int
        # print(decay_scores)
        # NMS_sorted_scores_list.append(sorted_scores[decay_order])
        # NMS_sorted_cate_label_list.append(sorted_cate[decay_order])
        # ori_h, ori_w = ori_size[0], ori_size[1]
        # ins_h, ins_w = ins_pred_img.shape[1], ins_pred_img.shape[2]
        # nms_sorted_ins = sorted_ins[decay_order]  # (keep_instance, ori_H/4, ori_W/4), float
        # nms_sorted_ins = torch.nn.functional.interpolate(nms_sorted_ins.view(1, keep_instance, ins_h, ins_w),
        #                                                  (ori_h, ori_w))  # (1, keep_instance, ori_H, ori_W), float
        # NMS_sorted_ins_list.append(nms_sorted_ins[0])
        # print(NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list)
        # return NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list

        pass

    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS

        # #compute IOU
        num_mask = len(sorted_scores)
        flatten_mask = torch.flatten(sorted_ins,start_dim=1)
        intersection_mat = flatten_mask @ flatten_mask.T
        area_mat = torch.sum(flatten_mask, dim=1)
        union_mat = area_mat + area_mat.T - intersection_mat

        # not sure why this but suggested in pseudocode, scores on sorted, uppertrianguler keeps the elements i<j
        # , which indicates s_i > s_j, thus satisfy the constrint
        iou_mat = intersection_mat/union_mat
        iou_mat = torch.triu(iou_mat,diagonal=1)

        iou_max,_ = torch.max(iou_mat, dim=0)
        ious_max = iou_max.expand(num_mask,num_mask).T


        # # alternative approach
        # iou_mat = iou_mat - torch.eye(len(iou_mat)) #shape: n_act x n_act
        # iou_max = torch.max(iou_mat,dim=1).expand(num_mask,num_mask).T # max iou between mask_i and the rest of masks

        if method == 'gauss':
            decay_mat = torch.exp(-(iou_mat ** 2 - ious_max ** 2 ) / gauss_sigma)
        else: #linear function
            decay_mat = (1-iou_mat) / (1-iou_max)

        #this line is not in psudocode but used to check score condition
        # decay_mat
        decay,_ = torch.min(decay_mat,dim = 0) #shape: (n_act,)
        decay_scores = decay * sorted_scores
        return decay_scores

        ## Todo: Reference code
        # N = len(sorted_ins)
        # sorted_ins = sorted_ins.reshape(N, -1)
        # # pre-compute the IoU matrix: (N, N)
        # intersection = torch.mm(sorted_ins, sorted_ins.T)
        # areas = sorted_ins.sum(dim=1).expand(N, N)
        # union = areas + areas.T - intersection + 1e-8
        # test = union.cpu().numpy()
        # ious = (intersection / union).triu(diagonal=1)
        # # max IoU for each: (N, N)
        # ious_cmax = ious.max(dim=0).values
        # ious_cmax = ious_cmax.expand(N, N).T
        #
        # # Matrix NMS: (N, N)
        # if method == 'gauss':
        #     decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        # else:  # linear
        #     decay = (1 - ious) / (1 - ious_cmax)
        #
        # # decay factor: (N,)
        # decay = torch.min(decay, dim=0).values
        #
        # return sorted_scores * decay


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
                plt.show()

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
        # ## TODO: Plot predictions
        # # loop through imgs in one batch
        for i in range(len(img)):
            fig, ax = plt.subplots(1)
            image = img[i]
            image = image.permute(1, 2, 0)
            ax.imshow(image)

            cate = NMS_sorted_cate_label_list[i]
            ins = NMS_sorted_ins_list[i]

            print('cate\n',cate)

            for ind in range(len(ins)):
                # if cate[ind] != 3:
                obj = ins[ind].numpy()
                obj = np.ma.masked_where(obj < 0.5, np.ones_like(obj))
                ax.imshow(obj.squeeze(),color_list[cate[ind]-1], alpha=0.7)

            fig.savefig("./testfig/infer_" + str(iter_ind * len(img) + i) + ".png")
            plt.show()
        # bz = len(img)
        # for i in range(bz):
        #     plt.figure()
        #     cur_img = img[i]
        #     plt.imshow(torch.clamp(cur_img, 0, 1).cpu().detach().permute(1, 2, 0).numpy())
        #     score = NMS_sorted_scores_list[i]  # (keep_instance,), float
        #     cate = NMS_sorted_cate_label_list[i]  # (keep_instance,), int
        #     ins = NMS_sorted_ins_list[i]  # (keep_instance, ori_H, ori_W)
        #     for index in range(len(ins)):
        #         if cate[index] == 3: continue
        #         item = ins[index].cpu().detach().numpy()
        #         item = np.ma.masked_where(item < 0.5, np.ones_like(item))
        #         plt.imshow(item, alpha=0.8, cmap=color_list[cate[index]])
        #     plt.savefig('Infer_{}'.format(iter_ind * bz + i))
        #     plt.show()

    def evaluation(self, NMS_sorted_scores_list, NMS_sorted_cate_label_list,
                   NMS_sorted_ins_list, ins_gts_list, cate_gts_list):
        '''
        :param NMS_sorted_scores_list:
        :param NMS_sorted_cate_label_list: shape:(bz, instance_number, 1)
        :param NMS_sorted_ins_list: shape:(bz, instance_number, ori_h, ori_w)
        :param ins_gts_list: (bz, instance_number)
        :param cate_gts_list: (bz, instance_number, ori_h, ori_w)
        :return:
        '''

        ## 0->pedestrian, 1->traffic, 2->car
        match = {0: [], 1: [], 2: []}  # prediction &grnd truth same class and IoU > 0.5.
        scores = {0: [], 1: [], 2: []}  # track of the confidence score for every prediction
        trues = {0: 0, 1: 0, 2: 0}
        print(NMS_sorted_cate_label_list, NMS_sorted_ins_list, ins_gts_list, cate_gts_list)

        # ## create selection mask from gnd truth label
        # labelMask = out_NMS[:, 0, :, :] != 0
        # trueLMask = labels[:, 0, :, :] != 0
        # clsMask = torch.argmax(out_NMS[:, 5:, :, :], dim=1)
        #
        # pedeMask = torch.bitwise_and(clsMask == 0, labelMask)
        # trafficMask = torch.bitwise_and(clsMask == 1, labelMask)
        # carMask = torch.bitwise_and(clsMask == 2, labelMask)
        #
        # masks = [pedeMask, trafficMask, carMask]
        # ## select
        # for cls in range(3):
        #     mask = masks[cls]
        #     ## check if empty selelct:
        #     total = torch.sum(mask.int()).item()
        #     if total != 0:
        #         score = out_NMS[:, 0][mask]
        #         scores[cls] = score.cpu()
        #         ## slise out all cls prediction
        #         output_candi = []  ## should be shape(#response,8)
        #         label_candi = []  ## should be shape(#response,8)
        #
        #         ## temp for loop deal with batch slice
        #         for b in range(len(out_NMS)):
        #             if b == 0:
        #                 output_candi = torch.transpose(out_NMS[b][:, mask[b]], 0, 1)
        #                 label_candi = torch.transpose(labels[b][:, mask[b]], 0, 1)
        #                 continue
        #             output_candi = torch.cat((output_candi, torch.transpose(out_NMS[b][:, mask[b]], 0, 1)), 0)
        #             label_candi = torch.cat((label_candi, torch.transpose(labels[b][:, mask[b]], 0, 1)), 0)
        #
        #         ## check if label are same class
        #         pair_label = (torch.argmax(output_candi[:, 5:], dim=1) - torch.argmax(label_candi[:, 5:], dim=1)) == 0
        #
        #         ## make box & calculate iou
        #         box1 = (output_candi[:, 1], output_candi[:, 2], output_candi[:, 3], output_candi[:, 4])
        #         box2 = (label_candi[:, 1], label_candi[:, 2], label_candi[:, 3], label_candi[:, 4])
        #         iou = IOU(box1, box2) > 0.5
        #
        #         ## calulate match
        #         match[cls] = torch.bitwise_and(pair_label, iou).cpu().int()
        #
        #     ## get true for this class
        #     trueClsMask = torch.bitwise_and(torch.argmax(labels[:, 5:, :, :], dim=1) == cls, trueLMask)
        #     trues[cls] = torch.sum(trueClsMask.int()).cpu().item()
        #     # pdb.set_trace()

        return match, scores, trues


        pass

    def average_precision(self,MATCH_cls, SCORES_cls, TRUES_cls):

        '''
          match: shape (1,#) torch
          score: shape (1,#) torch
          total: number
          '''
        from sklearn.metrics import auc
        ## please fill in the appropriate arguements
        ## compute the average precision as mentioned in the PDF.
        ## it might be helpful to use - from sklearn.metrics import auc to compute area under the curve.

        # match= match.numpy()
        # score=score.numpy()
        match = np.asarray(match)
        score = np.asarray(score)
        maximum_score = np.max(score)
        ln = np.linspace(0.6, maximum_score, 100)
        precision_mat = np.zeros((101))
        recall_mat = np.zeros((101))

        for i, th in enumerate(ln):
            match_th_mask = score > th
            TP = np.sum(match[match_th_mask])
            total_positive = len(match[match_th_mask])
            precision = 1
            if total_positive > 0:
                precision = TP / total_positive

            recall = 1
            total_trues = total
            if total_trues > 0:
                recall = TP / total_trues

            precision_mat[i] = precision
            recall_mat[i] = recall

        recall_mat[100] = 0
        precision_mat[100] = 1
        sorted_ind = np.argsort(recall_mat)
        sorted_recall = recall_mat[sorted_ind]
        sorted_precision = precision_mat[sorted_ind]

        area = auc(sorted_recall, sorted_precision)
        # plt.plot(sorted_recall, sorted_precision)
        # plt.show()
        # pdb.set_trace()

        return area



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
        # print(torch.where(ins_ind_gts_list[0][5] == True))
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)


        #testing the loss function here:
        loss_test = solo_head.loss(cate_pred_list,ins_pred_list,ins_gts_list,ins_ind_gts_list,cate_gts_list)
        print('tested loss function',loss_test)


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

