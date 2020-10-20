import solo_head
from dataset import *
import torch

def test_loss(solo_head_net):
    focal_input_path = "./data/focal_input.npz"
    focal_output_path = "./data/focal_output.npz"
    dice_in_path = "./data/dice_input.npz"
    dice_gt_path = "./data/dice_output.npz"

    focal_in = np.load(focal_input_path, allow_pickle= True)
    cate_preds = torch.from_numpy(focal_in['cate_preds'])
    cate_gts = torch.from_numpy(focal_in['cate_gts'])
    focal_out_gt = np.load(focal_output_path,allow_pickle= True)
    cate_loss = focal_out_gt['cate_loss']

    dice_in = np.load(dice_in_path,allow_pickle=True)
    dice_out = np.load(dice_gt_path,allow_pickle=True)
    print(dice_in.files)
    print(dice_out.files)
    print((dice_in['ins_preds']).shape)
    # print(dice_in['ins_gts'])
    ins_preds = dice_in['ins_preds']
    ins_gts = dice_in['ins_gts']
    ins_loss = dice_out['mask_loss']



    gt = torch.tensor([[0,0,1,],[0,1,1],[0,0,1]])
    gt = torch.unsqueeze(gt, dim=0)
    pred = torch.tensor([[0.2,0.3,0.9],[0.6,0.5,0.7],[0.4,0,0.2]])
    pred = torch.unsqueeze(pred,dim=0)
    dmask = solo_head_net.DiceLoss(pred,gt)

    # print(ins_preds)
    # print(ins_gts)
    #
    # dice_loss_test = solo_head_net.DiceLoss(ins_preds,ins_gts)
    # print('')
    # print('Dice loss input', dice_loss_test)
    dice_loss_sum = 0
    N_total = 0
    num_fpn = 5
    for i in range(num_fpn):
        dice_loss_sum += solo_head_net.DiceLoss(torch.from_numpy(ins_preds[i]), torch.from_numpy(ins_gts[i]))
        N_total += ins_preds[i].shape[0]
    mask_loss = dice_loss_sum / N_total
    print('Mask loss calculated', mask_loss*3)
    print('Mask loss gt', ins_loss)
    # print('Dice loss', dmask)

    simple_focal_pred = torch.tensor([[0.8,0.3,0.6],[0.1,0.2,0.5],[0.7,0.4,0.4]])
    # print('simple_focal_pred',simple_focal_pred)
    simple_focal_gt = torch.tensor([1,3,1])

    focal_loss_simple = solo_head_net.FocalLoss(simple_focal_pred,simple_focal_gt)
    focal_loss_test = solo_head_net.FocalLoss(cate_preds,cate_gts)
    print('Calculated focal loss', focal_loss_test)
    print('Ground Truth focal loss',cate_loss)
    print('Focal loss simple', focal_loss_simple)



    print('completed test')

def test_postprocess():
    pass

if __name__ == '__main__':
    solo_head_net = solo_head.SOLOHead(num_classes=4)
    test_loss(solo_head_net)