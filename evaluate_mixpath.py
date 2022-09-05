import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from mixpath_2 import Sub_SuperNetwork
from model_search import SuperNetwork
import scipy.io as scio#从fpn文件中model导入fpn类
from segmentation_evaluation import validate
from mat2pic import GeneralDataset, TestDataset, trans_separate
from model_init import weights_init, weights_init_without_kaiming
from scheduler import WarmupMultiStepLR
model_path = 'mixpath_fpn_1/data/fpn.pth.190'
# save_path='/mnt/sunjialiang/'
device = torch.device('cuda')

# choice = {
#     0: {'conv': [3], 'rate': 0},
#     1: {'conv': [3, 1], 'rate': 1},
#     2: {'conv': [0], 'rate': 1},
#     3: {'conv': [1, 0], 'rate': 1},
#     4: {'conv': [0, 3], 'rate': 0},
#     5: {'conv': [1], 'rate': 0},
#     6: {'conv': [3], 'rate': 0},
#     7: {'conv': [3], 'rate': 0},
#     8: {'conv': [3, 1], 'rate': 1},
#     9: {'conv': [1, 2], 'rate': 0},
#     10: {'conv': [0, 2], 'rate': 0},
#     11: {'conv': [2, 3], 'rate': 0}}
# choice = {
#     0: {'conv': [0], 'rate': 0},
#     1: {'conv': [0], 'rate': 0},
#     2: {'conv': [0], 'rate': 0},
#     3: {'conv': [0], 'rate': 0},
#     4: {'conv': [0], 'rate': 0},
#     5: {'conv': [0], 'rate': 0},
#     6: {'conv': [0], 'rate': 0},
#     7: {'conv': [0], 'rate': 0},
#     8: {'conv': [0], 'rate': 0},
#     9: {'conv': [0], 'rate': 0},
#     10: {'conv': [0], 'rate': 0},
#     11: {'conv': [0], 'rate': 0}}
# choice = {
#  0: {'conv': [0, 3], 'rate': 0},
#  1: {'conv': [2], 'rate': 1},
#  2: {'conv': [0, 1], 'rate': 0},
#  3: {'conv': [1], 'rate': 1},
#  4: {'conv': [3], 'rate': 0},
#  5: {'conv': [0, 2], 'rate': 0},
#  6: {'conv': [1], 'rate': 0},
#  7: {'conv': [3, 2], 'rate': 0},
#  8: {'conv': [3, 2], 'rate': 1},
#  9: {'conv': [2], 'rate': 0},
#  10: {'conv': [1], 'rate': 1},
#  11: {'conv': [3], 'rate': 0}}

# choice = {
#     0: {'conv': [0,1,2,3], 'rate': 1},
#     1: {'conv': [0,1,2,3], 'rate': 1},
#     2: {'conv': [0,1,2,3], 'rate': 1},
#     3: {'conv': [0,1,2,3], 'rate': 1},
#     4: {'conv': [0,1,2,3], 'rate': 1},
#     5: {'conv': [0,1,2,3], 'rate': 1},
#     6: {'conv': [0,1,2,3], 'rate': 1},
#     7: {'conv': [0,1,2,3], 'rate': 1},
#     8: {'conv': [0,1,2,3], 'rate': 1},
#     9: {'conv': [0,1,2,3], 'rate': 1},
#     10: {'conv': [0,1,2,3], 'rate': 1},
#     11: {'conv': [0,1,2,3], 'rate': 1}}

def single_evluation(path,model,choice):
    datadict = scio.loadmat(path)

    layout_map = np.zeros((200, 200))
    location = datadict['list'][0]

    for i in location:
      i = i - 1
      layout_map[i // 10 * 20:i // 10 * 20 + 20, (i % 10) * 20:(i % 10) * 20 + 20] = np.ones((20, 20))
    heat_map = (datadict['u'] - 260) / 100

    layout_map, heat_map = trans_separate(layout_map, heat_map, (200, 200))
    layout_map = layout_map.unsqueeze(0).to(device)

    # start = time.time()
    # for i in range(100):
    preds = model(layout_map)
    # end = time.time()
    # print(end - start)

    heat_map = heat_map.unsqueeze(0).to(device)
    # preds[:, :, 20:30, 0:5] = heat_map[:, :, 20:30, 0:5]

    # sequence = preds.flatten().cpu().detach().numpy()
    # sequence_y = heat_map.flatten().cpu().detach().numpy()
    # error = sequence- sequence_y
    # print('--------------')
    # print(np.max(error), np.min(error))
    # print('--------------')
    # index = np.argmax(sequence)
    # print(sequence[index] - sequence_y[index])
    pred_numpy = (preds.cpu().detach().numpy()[0, 0, :, :]) * 100 + 260
    # scio.savemat(save_path + '344' + '.mat', {'predu': pred_numpy})
    loss = F.l1_loss(preds, heat_map, reduction='mean').item()
    # if save_img:
    #     cv2.imwrite("layout_real.jpg", layout_real)
    #     cv2.imwrite("layout_pred.jpg", layout_pred)
    # preds = real_u(preds)
    # return loss , preds
    return loss, pred_numpy, np.max(pred_numpy)


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
model.cuda()
model.load_state_dict(torch.load(model_path, map_location=device))

dataset_test = TestDataset(trans_separate, resize_shape=(50, 50))
test_queue = torch.utils.data.DataLoader(
  dataset_test, batch_size=8, shuffle=False, pin_memory=True, num_workers=2)
print("Model Validation")
path = '/mnt/jfs/sunjialiang/mixpath_test/train/44.mat'
with torch.no_grad():
  model.eval()

  # mae, var, max_value = validate(model, test_queue, choice,max_iter=-1, flag_detail=False)
  # print("1-MAE:", round(mae * 100, 4), 'var:', round(var * 100, 4),
  #       'max_error:', round(max_value, 4))
  # for i in range(1000):
  loss, pred_numpy, max_error=single_evluation(path,model,choice)
  # print(loss, pred_numpy, max_error)
  print(loss,pred_numpy, max_error)
