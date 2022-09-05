import scipy.io as scio
import numpy as np
import os
from torch.utils.data import Dataset
import torch
import cv2


class MyDataset(Dataset):
    """
    TODO: 考虑 -0.5 ~ 0.5
    """
    def __init__(self, root, transform_func, resize_shape):
        self.root = root
        self.mats = sorted(os.listdir(self.root))
        self.transform_func = transform_func
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.mats)

    def __getitem__(self, item):
        mat = self.mats[item]
        layout_map = np.zeros((200, 200))
        datadict = scio.loadmat(self.root + mat)
        location = datadict['list'][0]

        for i in location:
            i = i - 1
            layout_map[i // 10 * 20:i // 10 * 20 + 20, (i % 10) * 20:(i % 10) * 20 + 20] = np.ones((20, 20))

        heat_map = (datadict['u'] - 260) / 100

        return self.transform_func(layout_map, heat_map, self.resize_shape)
    # def __getitem__(self, item):
    #     mat = self.mats[item]
    #     layout_map = np.zeros((200, 200))
    #     datadict = scio.loadmat(self.root+mat)
    #     location = datadict['list'][0]
    #
    #     for i in location:
    #         i = i - 1
    #         layout_map[i // 10 * 20:i // 10 * 20 + 20, (i % 10) * 20:(i % 10) * 20 + 20] = np.ones((20, 20))
    #
    #     heat_map = (datadict['u']-260)/100
    #
    #     return self.transform_func(layout_map, heat_map, self.resize_shape)


class GeneralDataset(MyDataset):
    def __init__(self, transform_func, resize_shape):
        # super(GeneralDataset, self).__init__('/data/nfsdata/cv/tmp/hushuxian/datam/', transform_func, resize_shape)
        # super().__init__('/data/nfsdata/cv/data/intelligent-design/layout-design/train_shape/', transform_func, resize_shape)
        # super().__init__('/data/nfsdata/cv/data/intelligent-design/layout-design/train/', transform_func, resize_shape)
        super().__init__('/mnt/jfs/sunjialiang/mixpath_test/train/', transform_func, resize_shape)

class TestDataset(MyDataset):
    def __init__(self, transform_func, resize_shape):
        # super(TestDataset, self).__init__('/data/nfsdata/cv/tmp/hushuxian/datat/', transform_func, resize_shape)
        # super().__init__('/data/nfsdata/cv/data/intelligent-design/layout-design/test_shape/', transform_func, resize_shape)
        # super().__init__('/data/nfsdata/cv/data/intelligent-design/layout-design/test/', transform_func, resize_shape)
        super().__init__('train/', transform_func, resize_shape)
def trans_stack(layout_map, heat, resize_shape):
    res = np.vstack((layout_map, heat))
    res = cv2.resize(res, resize_shape)
    res= np.expand_dims(res, 0)
    return torch.from_numpy(res.astype(np.float32))


def trans_concat(layout_map, heat_map, resize_shape):
    res = np.array([cv2.resize(layout_map, resize_shape), cv2.resize(heat_map, resize_shape)])
    return torch.from_numpy(res.astype(np.float32))

def trans_separate(layout_map, heat_map, resize_shape):
    layout_map = np.expand_dims(cv2.resize(layout_map, resize_shape), 0)
    heat_map = np.expand_dims(cv2.resize(heat_map, resize_shape), 0)
    return torch.from_numpy(layout_map.astype(np.float32)), torch.from_numpy(heat_map.astype(np.float32))


if __name__ == "__main__":
    dataset = GeneralDataset(trans_stack, (64, 64))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for it, images in enumerate(data_loader):
        print(images.shape)
        print(torch.max(images), torch.min(images))
        break
