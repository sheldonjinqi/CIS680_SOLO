import solo_head
from dataset import *

focal_input_path = "./data/focal_input.npz"
focal_output_path = "./data/focal_output.npz"

focal_in = np.load(focal_input_path, allow_pickle= True)
cate_preds = torch.from_numpy(focal_in['cate_preds'])
cate_gts = torch.from_numpy(focal_in['cate_gts'])
focal_out_gt = np.load(focal_output_path,allow_pickle= True)
cate_loss = focal_out_gt['cate_loss']

print('cate_preds',cate_preds.shape)
print('cate_gts',cate_gts.shape)



solo_head_net = solo_head.SOLOHead(num_classes=4)
pred = torch.tensor([[0,0,1,],[0,1,1],[0,0,1]])
gt = torch.tensor([[0.2,0.3,0.9],[0.6,0.5,0.7],[0.4,0,0.2]])
dmask = solo_head_net.DiceLoss(pred,gt)
print('Dice loss', dmask)

simple_focal_pred = torch.tensor([[0.8,0.3,0.6],[0.1,0.2,0.5],[0.7,0.4,0.4]])
# print('simple_focal_pred',simple_focal_pred)
simple_focal_gt = torch.tensor([1,3,1])
focal_loss_simple = solo_head_net.FocalLoss(simple_focal_pred,simple_focal_gt)
focal_loss_test = solo_head_net.FocalLoss(cate_preds,cate_gts)
print('Calculated focal loss', focal_loss_test)
print('Ground Truth focal loss',cate_loss)
print('Focal loss simple', focal_loss_simple)

print('completed test')

