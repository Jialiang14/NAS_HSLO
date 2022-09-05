import os
import argparse
import torch
from utils import *
import torch.nn as nn
from tqdm import tqdm
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from mixpath_2 import Sub_SuperNetwork
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from segmentation_2 import validate
# from mat2pic_2 import GeneralDataset_2, TestDataset_2, trans_separate
from mat2pic import GeneralDataset, TestDataset, trans_separate
from model_init import weights_init, weights_init_without_kaiming
from torchstat import stat
from scheduler import WarmupMultiStepLR
from torch.optim.lr_scheduler import MultiStepLR

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Use CPU")
        device = torch.device('cpu')
    else:
        print("Use GPU")

    batch_size = 32
    max_epochs = 200
    model_save_interval = 10
    model_valid_interval = 10
    LEARNING_RATE = 1e-2
    LOAD_PRETRAIN = False
    kernel = [3, 5, 7, 9]
    m_path = 2
    layers = 12

    # choice = {
    #     0: {'conv': [0], 'rate': [0]},
    #     1: {'conv': [0], 'rate': [0]},
    #     2: {'conv': [0], 'rate': [0]},
    #     3: {'conv': [0], 'rate': [0]},
    #     4: {'conv': [0], 'rate': [0]},
    #     5: {'conv': [0], 'rate': [0]},
    #     6: {'conv': [0], 'rate': [0]},
    #     7: {'conv': [0], 'rate': [0]},
    #     8: {'conv': [0], 'rate': [0]},
    #     9: {'conv': [0], 'rate': [0]},
    #     10: {'conv': [0], 'rate': [0]},
    #     11: {'conv': [0], 'rate': [0]}}

    # choice = {
    #     0: {'conv': [0,1,2,3], 'rate': [1]},
    #     1: {'conv': [0,1,2,3], 'rate': [1]},
    #     2: {'conv': [0,1,2,3], 'rate': [1]},
    #     3: {'conv': [0,1,2,3], 'rate': [1]},
    #     4: {'conv': [0,1,2,3], 'rate': [1]},
    #     5: {'conv': [0,1,2,3], 'rate': [1]},
    #     6: {'conv': [0,1,2,3], 'rate': [1]},
    #     7: {'conv': [0,1,2,3], 'rate': [1]},
    #     8: {'conv': [0,1,2,3], 'rate': [1]},
    #     9: {'conv': [0,1,2,3], 'rate': [1]},
    #     10: {'conv': [0,1,2,3], 'rate': [1]},
    #     11: {'conv': [0,1,2,3], 'rate': [1]}}

    choice = {
        0: {'conv': [2, 1, 0, 3], 'rate': [0]},
        1: {'conv': [3, 0, 2], 'rate': [1]},
        2: {'conv': [0, 2, 1], 'rate': [0]},
        3: {'conv': [1, 2, 0, 3], 'rate': [1]},
        4: {'conv': [1], 'rate': [0]},
        5: {'conv': [0, 3, 1], 'rate': [1]},
        6: {'conv': [0, 3, 2], 'rate': [0]},
        7: {'conv': [0], 'rate': [0]},
        8: {'conv': [2, 3, 0, 1], 'rate': [1]},
        9: {'conv': [0, 3, 1], 'rate': [0]},
        10: {'conv': [1, 2, 0], 'rate': [0]},
        11: {'conv': [2, 1], 'rate': [1]}}

    model = Sub_SuperNetwork(choice, shadow_bn=False, layers=12, classes=10, final_upsampling=4)
    model = model.to(device)
    project_path = 'mixpath_fpn_1'
    model_path = os.path.join(project_path, 'data', 'fpn.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    log_dir = os.path.join(project_path, 'log', 'fpn')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    dataset = GeneralDataset(trans_separate, resize_shape=(200, 200))  # 生成数据集，trans_separate是什么参数
    dataset_test = TestDataset(trans_separate, resize_shape=(200, 200))

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_test, batch_size=16, shuffle=False, drop_last=True)

    print("model path:", model_path)

    if LOAD_PRETRAIN and os.path.exists(model_path):
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("model initiated with", model_path)
    else:
        # model.backbone.apply(weights_init)# TODO@LYC: Init Header
        # model.head.apply(weights_init_without_kaiming) # not very effective
        print("model initiated without pretrain")
    for p in model.parameters():
        p.requires_grad = True

    print("\tLearning Rate:", LEARNING_RATE)
    print("\tBatch Size:", batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=[3125, 9375, 15625])
    # scheduler = WarmupMultiStepLR(optimizer, milestones=[], warmup_iters=len(train_loader))

    for epoch in range(max_epochs):
        for it, images in enumerate(train_loader):
            layout_image = images[0].to(device)
            heat_image = images[1].to(device)
            m = model(layout_image)  #这里是热场到布局图的映射
            loss = F.l1_loss(m, heat_image)
            del heat_image, layout_image
            torch.cuda.empty_cache()

            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("\tEpoch[{}/{}] Iters[{}/{}] Loss: {:.3f}".format(
                epoch + 1, max_epochs,
                it, len(train_loader),
                loss.item()*1e5),
            )

        if epoch % model_valid_interval == 0:
            model.eval()
            print("Model Validation")
            with torch.no_grad():
                mae, var, max_value = validate(model, valid_loader, max_iter=-1, flag_detail=False)
                print("1-MAE:", round(mae * 100, 4),  'var:', round(var * 100, 4),
                      'max_error:', round(max_value, 4))
            model.train()

        if epoch % model_save_interval == 0:
            torch.save(model.state_dict(), model_path+"."+str(epoch))
            print("Model Saved:", model_path+"."+str(epoch))
