# CIS680_SOLO
Implementation of SOLO Algorithm for CIS680

The code is structured same as the instruction code: 

- for checking dataset plot, run 'dataset.py' 
- for checking plot feature pyramid recovery, run 'solo_head.py'
- for training the model, run 'main_train.py'
- to get infer figures, run 'main_infer.py', set function infer(solo_head,resnet50_fpn, test_loader, device=device, mode='infer')
- to test mAp, run 'main_infer.py', set function infer(solo_head,resnet50_fpn, test_loader, device=device, mode='map')
