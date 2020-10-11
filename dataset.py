## Author: Lishuo Pan 2020/4/18

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # TODO: load dataset, make mask list
        self.path = path
        imgs_path, masks_path, labels_path, bboxes_path = self.path
        self.imgs_data = h5py.File(imgs_path,'r').get('data')
        input_masks_data = h5py.File(masks_path,'r').get('data')
        self.masks_data = []
        self.labels_data = np.load(labels_path, allow_pickle=True,encoding="latin1")
        self.bboxes_data = np.load(bboxes_path, allow_pickle=True)
        idx = 0
        #linking the mask and image
        for i in range(len(self.imgs_data)):
            num_mask = len(self.labels_data[i])
            self.masks_data.append(input_masks_data[idx:num_mask+idx,:,:])
            idx += num_mask
        # self.masks_data = np.asarray(self.masks_data)

    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__

        #normalize the pixel value to [0,1]
        img =  torch.tensor(self.imgs_data[index].astype(np.float32),dtype = torch.float)
        bbox = torch.tensor(self.bboxes_data[index],dtype = torch.float)
        label = torch.tensor(self.labels_data[index],dtype = torch.float)
        mask = torch.tensor(self.masks_data[index].astype(np.float32),dtype = torch.float)
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img,mask,bbox)
        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox
    def __len__(self):
        return len(self.imgs_data)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess
        #normalize the pixel value to [0,1]
        img = torch.tensor((img / 255.0), dtype=torch.float)

        # rescaling
        img = F.interpolate(img, size=1066)
        img=img.permute(0, 2, 1)
        img=F.interpolate(img, size=800)
        img=img.permute(0, 2, 1)
        #normalize each channel
        # print('mean before preprocess',torch.mean(img,(1,2)),torch.std(img,(1,2)))
        normalize = transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225),inplace = False)
        img = normalize(img)
        img = F.pad(img, (11, 11))
        # print('max after preprocess', torch.max(img))

        # same process to mask
        mask = F.interpolate(mask, size=1066)
        mask = mask.permute(0, 2, 1)
        mask = F.interpolate(mask, size=800)
        mask = mask.permute(0, 2, 1)
        mask = F.pad(mask, (11, 11))


        #what is this?
        bbox = bbox * 8/3
        bbox[:,0] += 11
        bbox[:,2] += 11

        # exit()
        # check flag
        # print(bbox.shape[0], mask.squeeze(0).shape[0])
        # print(bbox,mask)
        assert img.shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.shape[0]
        return img, mask, bbox


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        # TODO: collect_fn
        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []
        for transed_img, label, transed_mask, transed_bbox in batch:
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)

        # return torch.stack([transed_img_list,label_list, transed_mask_list,transed_bbox_list], dim=0)
        return torch.stack(transed_img_list,dim=0),\
               label_list, \
               transed_mask_list,\
               transed_bbox_list

    def loader(self):
        # TODO: return a dataloader

        return DataLoader(self.dataset, collate_fn=self.collect_fn, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)
    # print(dataset.imgs_data.shape)
    # print(dataset.labels_data[:2])
    # print(dataset.masks_data.shape)
    # print(dataset.bboxes_data.shape)
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
    batch_size = 5
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        # print(len([data[i] for i in range(len(data))]))
        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]


        # plot the origin img
        for i in range(batch_size):
            ## TODO: plot images with annotations
            # plot img
            image = img[i]
            im = np.zeros((800, 1088, 3))
            im[:, :, 0] = image[0, :, :]
            im[:, :, 1] = image[1, :, :]
            im[:, :, 2] = image[2, :, :]
            fig, ax = plt.subplots(1)
            ### Display the image ###
            ax.imshow(np.squeeze(im))

            # plot bounding box
            for box in bbox[i]:
                # Create a Rectangle patch
                # print(box)
                x,y = box[0], box[1]
                w = (box[2]-box[0])
                h = (box[3]-box[1])
                rect = patches.Rectangle((x,y), w, h,
                                         linewidth=1, edgecolor='r', facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)

            # plot mask
            print('image: ',i)
            print('num of mask', len(mask[i]))
            for j,msk in enumerate(mask[i]):
                cls = label[i][j]
                print(cls)
                msk = np.ma.masked_where(msk == 0, msk)
                print(msk.shape)
                plt.imshow(msk, mask_color_list[int(cls)], interpolation='none', alpha=0.7)

            # plt.savefig("./testfig/visualtrainset"+str(iter)+"_"+ str(i)+".png")
            plt.show()
            # print(label[i].shape,mask[i].shape,bbox[i].shape)

        if iter == 0:
            break

