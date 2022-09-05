
import torch
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from numpy import var
import cv2
from post_process import post_get_top_20, norm_label


def __predict(input, label, net, choice,flag_cost = False):

    predict = net(input,choice)
    if not flag_cost:
        return predict, None

    cost = F.binary_cross_entropy(predict, label, reduction="mean")
    return predict, cost

def metric(pred, label):
    mae = F.l1_loss(pred, label, reduction='mean').item()
    # acc = 0
    # for i in range(pred.size()[0]):
    #     acc += np.mean(post_get_top_20(pred[i,0,:,:]) == norm_label(label[i,0,:,:]))
    # acc /= pred.size()[0]
    # pred_20_index = F.interpolate(pred, size=(10, 10), mode='bilinear').view(pred.size(0), -1).topk(20)[1]
    # label_20 = F.interpolate(label, size=(10, 10), mode='bilinear').view(label.size(0), -1)
    # acc = 1.4 - 2 * (label_20.scatter_(1, pred_20_index, 1)).mean()

    return 1 - mae

def validate(net, val_loader, choice, max_iter= -1, flag_detail=True, save_img = False):
    """
    从validation set中取数个batch用于验证
    :param net:       模型
    :param val_loader:  验证集的dataloader
    :param max_iter:  本次验证要取的batch数目
    :return:
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    acc_total = 0
    mae_total = 0

    # 原始字符串
    val_iter = iter(val_loader)

    max_iter = min(max_iter, len(val_loader)) if max_iter > 0 else len(val_loader)

    if save_img:
        assert valid_loader.batch_size == 1
        dataset_mats = valid_loader.dataset.mats

        import time
        project_path = '/home/chenxianqi/layout_design/sunjialiang/fpn_master/fpn-master/model'
        save_path = os.path.join(project_path, "data", "evaluation", "combine_{}".format(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))))
        print("Saving Image in", save_path)
        os.makedirs(save_path, exist_ok=True)
    MAEE = np.arange(max_iter, dtype = np.float)
    for i in tqdm(range(max_iter)):
        images = val_iter.next()
        layout_image = images[0].to(device)
        heat_image = images[1].to(device)

        preds, _ = __predict(layout_image, heat_image, net, choice)

        mae = metric(preds, heat_image)

        if save_img:
            pred_numpy = cv2.resize((preds.cpu().numpy()[0, 0, :, :] * 255).astype("uint8"), (10, 10))
            layout_numpy = cv2.resize((layout_image.cpu().numpy()[0, 0, :, :] * 255).astype("uint8"), (10, 10))
            heat_numpy = (heat_image.cpu().numpy()[0, 0, :, :] * 255).astype("uint8")

            pred_numpy_200 = np.repeat(np.repeat(pred_numpy, 20, axis=0), 20, axis=1)
            layout_numpy_200 = np.repeat(np.repeat(layout_numpy, 20, axis=0), 20, axis=1)

            image_combine = np.concatenate([pred_numpy_200, heat_numpy, layout_numpy_200], axis=0)
            cv2.imwrite(os.path.join(save_path, "{}_{}.jpg".format(i, dataset_mats[i])), image_combine)

        if flag_detail:
            print("[{}/{}] {}".format(i, max_iter, acc))

        # acc_total += acc
        mae_total += mae
        MAEE[i] = (1-mae)*100

        # 展示一个结果
        # if flag_detail and i == 0:
            # cv2.imwrite()

    mae = mae_total / max_iter
    accuracy = acc_total / max_iter

    return mae, var(MAEE), np.max(MAEE)

if __name__ == "__main__":
    import os

    from mat2pic import TestDataset, GeneralDataset, trans_separate
    import sys
    sys.path.append("..")
    from model import fpn
    from torch.utils.data import DataLoader
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        print("Use CPU")
        device = torch.device('cpu')

    model = fpn().to(device)
    model_path = os.path.join(project_path, 'data', 'fpn.pth')

    dataset_test = TestDataset(trans_separate, resize_shape=(200, 200))
    valid_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=True)

    print("model path:", model_path)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    print("Model Validation")
    with torch.no_grad():
        mae, accuracy ,var = validate(model, valid_loader, max_iter=2000, flag_detail=True, save_img=True)
        print("1-MAE:", round(mae * 100, 4), 'Accuracy:', round(accuracy * 100, 4),'var:', round(accuracy * 100, 4))

