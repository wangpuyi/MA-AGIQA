import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from models.MA_AGIQA import MA_AGIQA
from config import Config
from utils.process import RandCrop, ToTensor, Normalize, five_point_crop
from utils.process import dataset_imagereward, dataset_agiqa, dataset_laion, dataset_general
from utils.process import RandRotation, RandHorizontalFlip
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
from performance import performance_fit


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def test_epoch(config, epoch, net, criterion, test_loader, output_csv=""):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []
        import csv
        csv_file = output_csv

        if output_csv:
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                # 写入列名
                writer.writerow(["name", "prediction", "gt"])

        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                labels = data['score']
                names = data['name_']
                tensor1 = data['tensor_1'].cuda()
                tensor2 = data['tensor_2'].cuda()

                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                x_d = five_point_crop(i, d_img=x_d, config=config)
                pred += net(x_d, tensor1=tensor1, tensor2=tensor2)

            pred /= config.num_avg_val

            if output_csv:
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # 写入数据
                    for i in range(len(pred)):
                        writer.writerow([names[i], pred[i].item(), labels[i].item()])
            

            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
        print('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
        test_srcc, test_plcc, test_krcc, test_rmse = performance_fit(labels_epoch, pred_epoch)

        return test_srcc, test_plcc, test_krcc, test_rmse


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    # dataset_name: ImageReward, ...
    config = Config({
        # dataset path
        "dataset_name": "AIGCQA_30k",
        "load_ckpt_path" : "output/models/original/ckpt_koniq10k.pt",

        # PIPAL
        "train_dis_path": "/mnt/IQA_dataset/PIPAL22/Train_dis/",
        "val_dis_path": "/mnt/IQA_dataset/PIPAL22/Val_dis/",
        "pipal22_train_label": "./data/PIPAL22/pipal22_train.txt",
        "pipal22_val_txt_label": "./data/PIPAL22/pipal22_val.txt",

        # KADID-10K
        "kadid10k_path": "/mnt/IQA_dataset/kadid10k/images/",
        "kadid10k_label": "./data/kadid10k/kadid10k_label.txt",

        # KONIQ-10K
        "koniq10k_path": "/mnt/IQA_dataset/1024x768/",
        "koniq10k_label": "./data/koniq10k/koniq10k_label.txt",

        # ImageReward
        "imagereward_path": "/data/wangpuyi_data/ImageRewardDB",
        "IR_label": "data/imagereward",

        #AGIQA
        "agiqa_path": "/data/wangpuyi_data/AGIQA-3K",
        "agiqa_label": "data/agiqa",  

        #AIGCQA_30k
        "AIGCQA_30k_path": "/data/wangpuyi_data/AIGCQA-30k",
        "AIGCQA_30k_label": "data/AIGCQA_30K_Image",  

        # LAION
        "laion_path": "/data/wangpuyi_data/home/jdp/simulacra-aesthetic-captions",
        "laion_label": "data/laion",

        # optimization
        "batch_size": 8,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 300,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 1, # if training koniq10k, num_avg_val is set to 1
        "num_workers": 8,
        
        # data
        "split_seed": 20,
        "train_keep_ratio": 1.0,
        "val_keep_ratio": 1.0,
        "crop_size": 224,
        "prob_aug": 0.7,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,
        
        # load & save checkpoint
        "model_name": "AGIQA-base_s20",
        "type_name": "AGIQA",
        "ckpt_path": "./output/models/",               # directory for saving checkpoint
        "log_path": "./output/log/",
        "log_file": ".log",
        "tensorboard_path": "./output/tensorboard/"
    })
    
    config.log_file = config.model_name + ".log"
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.type_name)
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.model_name)

    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name)
    config.ckpt_path = os.path.join(config.ckpt_path, config.model_name)

    config.log_path = os.path.join(config.log_path, config.type_name)

    if config.dataset_name == 'AGIQA':
        from data.AIGC_general import AIGCgeneral
        dis_train_path = config.agiqa_path
        dis_val_path = config.agiqa_path
        dis_test_path = config.agiqa_path
        test_names, test_labels = dataset_general(config.agiqa_label, 'test')
        Dataset = AIGCgeneral
    elif config.dataset_name == 'AIGCQA_30k':
        from data.AIGC_general import AIGCgeneral
        dis_train_path = config.AIGCQA_30k_path
        dis_val_path = config.AIGCQA_30k_path
        dis_test_path = config.AIGCQA_30k_path
        test_names, test_labels = dataset_general(config.AIGCQA_30k_label, 'all')
        Dataset = AIGCgeneral
    else:
        pass

    test_dataset = Dataset(
        dis_path=dis_test_path,
        labels=test_labels,
        pic_names=test_names,
        transform=transforms.Compose([RandCrop(patch_size=config.crop_size),
            Normalize(0.5, 0.5), ToTensor()]),
        keep_ratio=config.val_keep_ratio
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, drop_last=True, shuffle=True)
    epoch = 1

    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    # net = nn.DataParallel(net)
    ckpt_path = config.load_ckpt_path
    net.load_state_dict(torch.load(ckpt_path))
    net = net.cuda()
    criterion = torch.nn.MSELoss()
    test_srcc, test_plcc, test_krcc, test_rmse = test_epoch(config, epoch, net, criterion, test_loader)

    print('Testing SRCC %4.4f,\tPLCC %4.4f,\tKRCC %4.4f,\tRMSE %4.4f' % (test_srcc, test_plcc, test_krcc, test_rmse))