import os
import argparse
import torch
from utils import *
import torch.nn as nn
from tqdm import tqdm
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from model_search import SuperNetwork
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from segmentation_evaluation import validate
from mat2pic import GeneralDataset, TestDataset, trans_separate
from model_init import weights_init, weights_init_without_kaiming
from torchstat import stat
from scheduler import WarmupMultiStepLR
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Use CPU")
        device = torch.device('cpu')
    else:
        print("Use GPU")

    batch_size = 32
    max_epochs = 600
    model_save_interval = 10
    model_valid_interval = 10
    LEARNING_RATE = 1e-3
    LOAD_PRETRAIN = False
    kernel = [3, 5, 7, 9]
    m_path = 4
    layers = 12
    model = SuperNetwork(shadow_bn=False, layers=12, classes=10, final_upsampling=2)
    model = model.to(device)
    project_path = 'mixpath_supernet_kw'
    model_path = os.path.join(project_path, 'data', 'fpn.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    log_dir = os.path.join(project_path, 'log', 'fpn')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    dataset = GeneralDataset(trans_separate, resize_shape=(50, 50))  # 生成数据集，trans_separate是什么参数
    dataset_test = TestDataset(trans_separate, resize_shape=(50, 50))

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
    scheduler = WarmupMultiStepLR(optimizer, milestones=[], warmup_iters=len(train_loader))

    USE_NORMAL_LOSS = True # 截断大loss
    # input_one = (1,1,50,50)
    # stat(model,(input_one,choice))
    for epoch in range(max_epochs):
        if USE_NORMAL_LOSS:
            normal_loss = -1
        choice = random_choice(path_num=len(kernel), m=m_path, layers=layers)
        for it, images in enumerate(train_loader):
            layout_image = images[0].to(device)
            heat_image = images[1].to(device)
            m = model(layout_image, choice)  #这里是热场到布局图的映射
            loss = F.l1_loss(m, heat_image)
            del heat_image, layout_image
            torch.cuda.empty_cache()
            if USE_NORMAL_LOSS and epoch > 0:
                if normal_loss == -1:
                    normal_loss = loss.item()
                if loss.item() > normal_loss * 10:
                    writer.add_scalar('LOSS', -1, epoch * len(train_loader) + it)
                    continue
                normal_loss = normal_loss * 0.9 + loss.item() * 0.1

            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("\tEpoch[{}/{}] Iters[{}/{}] Loss: {:.3f}".format(
                epoch + 1, max_epochs,
                it, len(train_loader),
                loss.item()*1e5),
            )
            writer.add_scalar('LOSS', loss.item(), epoch * len(train_loader) + it)

        if epoch % model_valid_interval == 0:
            model.eval()
            print("Model Validation")
            with torch.no_grad():
                mae, var, max_value = validate(model, valid_loader,choice, max_iter=-1, flag_detail=False)
                print("1-MAE:", round(mae * 100, 4),  'var:', round(var * 100, 4),
                      'max_error:', round(max_value, 4))
            model.train()

            writer.add_scalar('1-MAE', mae, epoch)
            # writer.add_scalar('ACC', accuracy, epoch)

        if epoch % model_save_interval == 0:
            torch.save(model.state_dict(), model_path+"."+str(epoch))
            print("Model Saved:", model_path+"."+str(epoch))
