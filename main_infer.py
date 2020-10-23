'''
Main training process SOLO
Author: Chang Liu, Qi Jin
Date: 10/15/2020
'''
from main_train import *
import os

def load_model(net, optimizer, device,epoch_num=0):
    # path = os.path.join('./network_result1/', 'solo_epoch_li' + str(epoch_num))
    path = os.path.join('./network_result_21/', 'solo_epoch_' + str(epoch_num))
    path = os.path.join('', 'solo_epoch_' + str(epoch_num))
    checkpoint = torch.load(path,map_location=torch.device(device))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print('Model loaded')
    return net, optimizer

def infer(solo_head,resnet50_fpn, test_loader, device='cpu', mode='infer'):
    solo_head.eval()
    resnet50_fpn.eval()
    color_list = ["jet", "ocean", "Spectral"]
    MATCH = {0: [], 1: [], 2: []}
    SCORES = {0: [], 1: [], 2: []}
    TRUES = {0: 0, 1: 0, 2: 0}

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
                solo_head.PlotInfer(NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list,
                                    color_list, imgs, i)
                if i == 20:
                    exit()
            if mode=='map':
                match, scores, trues = solo_head.evaluation(NMS_sorted_ins_list,
                                                            NMS_sorted_cate_label_list,
                                                            NMS_sorted_scores_list,
                                                            mask_list,
                                                            label_list,)
                for c1 in range(3):
                    MATCH[c1] += match[c1]
                    SCORES[c1] += scores[c1]
                    TRUES[c1] += trues[c1]

        if mode=='map':
            Ap = 0
            Ap_list = np.zeros(3)
            count = 0
            mAp = 0
            for i in range(3):
                if (len(MATCH[i]) > 0):
                    Ap_list[i] = solo_head.average_percision(MATCH[i], SCORES[i], TRUES[i])
                    Ap += Ap_list[i]
                    count += 1
            if count > 0:
                mAp = Ap / count
            return mAp, Ap_list


def main_infer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ## load data
    train_loader, test_loader = load_data(batch_size=2)
    print('data loaded')

    ## initialize backbone and SOLO head
    resnet50_fpn = Resnet50Backbone().to(device)
    solo_head = SOLOHead(num_classes=4,device=device).to(device)  ## class number is 4, because consider the background as one category.

    ## initialize optimizer
    learning_rate = 1e-2  # for batch size 2
    import torch.optim as optim
    optimizer = optim.SGD(solo_head.parameters(),
                          lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    ## only check the final model we have, output example figures
    solo_head, optimizer = load_model(solo_head, optimizer, device, epoch_num=40)
    map, map_list = infer(solo_head,resnet50_fpn, test_loader, device=device, mode='infer')
    print('map, map_list ',map, map_list)

if __name__ == '__main__':
    main_infer()