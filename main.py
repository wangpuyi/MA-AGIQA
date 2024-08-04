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
from utils.process import split_dataset_kadid10k, split_dataset_koniq10k, dataset_imagereward, dataset_general
from utils.process import RandRotation, RandHorizontalFlip
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from performance import performance_fit

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.dataset_name + ".log")
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader):
        x_d = data['d_img_org'].cuda()
        labels = data['score']
        tensor1 = data['tensor_1'].cuda()
        tensor2 = data['tensor_2'].cuda()

        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()  
        pred_d = net(x_d, tensor1=tensor1, tensor2=tensor2)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p


def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                labels = data['score']
                tensor1 = data['tensor_1'].cuda()
                tensor2 = data['tensor_2'].cuda()
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                x_d = five_point_crop(i, d_img=x_d, config=config)
                pred += net(x_d, tensor1=tensor1, tensor2=tensor2)

            pred /= config.num_avg_val
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
        return np.mean(losses), rho_s, rho_p

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
    import argparse
    from omegaconf import OmegaConf
    setup_seed(20)

    parser = argparse.ArgumentParser(description='Dataset Understanding')
    parser.add_argument('--config', default='configs/base.yaml', help="config file")
    
    flags, unknown = parser.parse_known_args()

    cfg       = OmegaConf.load(flags.config)
    base      = OmegaConf.load('configs/base.yaml')
    config      = OmegaConf.merge(base, cfg)

    config.log_file = config.dataset_name + ".log"
    model_save_path = "checkpt/{}.pt".format(config.dataset_name)

    set_logging(config)
    logging.info(config)
    cpu_num = config.cpu_num
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    srcc_all = np.zeros(config.train_test_num, dtype=float)
    plcc_all = np.zeros(config.train_test_num, dtype=float)
    krcc_all = np.zeros(config.train_test_num, dtype=float)
    rmse_all = np.zeros(config.train_test_num, dtype=float)

    for i in range(config.train_test_num):
        if config.dataset_name == 'AIGCQA_20k' or config.dataset_name == 'AGIQA_3k':
            from data.AIGC_general import AIGCgeneral
            dis_train_path = config.data_path
            dis_val_path = config.data_path
            type_ = 'random'
            if type_ == 'random':
                train_names, train_labels, val_names, val_labels, test_names, test_labels = dataset_general(config.label_path, type_, config.dataset_name, set_seed=config.set_seed)
            Dataset = AIGCgeneral
        else:
            raise NotImplementedError
        
        # data load
        if config.dataset_name == 'AIGCQA_20k' or config.dataset_name == 'AGIQA_3k':
            train_dataset = Dataset(
                dis_path=dis_train_path,
                labels=train_labels,
                pic_names=train_names,
                transform=transforms.Compose([RandRotation(prob_aug=config.prob_aug), RandCrop(patch_size=config.crop_size), 
                    Normalize(0.5, 0.5), RandHorizontalFlip(prob_aug=config.prob_aug), ToTensor()]),
                keep_ratio=config.train_keep_ratio,
                tensor_root=config.tensor_root
            )
            val_dataset = Dataset(
                dis_path=dis_val_path,
                labels=val_labels,
                pic_names=val_names,
                transform=transforms.Compose([RandCrop(patch_size=config.crop_size),
                    Normalize(0.5, 0.5), ToTensor()]),
                keep_ratio=config.val_keep_ratio,
                tensor_root=config.tensor_root
            )
            test_dataset = Dataset(
                dis_path=dis_val_path,
                labels=test_labels,
                pic_names=test_names,
                transform=transforms.Compose([RandCrop(patch_size=config.crop_size),
                    Normalize(0.5, 0.5), ToTensor()]),
                keep_ratio=config.val_keep_ratio,
                tensor_root=config.tensor_root
            )

        logging.info('number of train scenes: {}'.format(len(train_dataset)))
        logging.info('number of val scenes: {}'.format(len(val_dataset)))

        # load the data
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
            num_workers=config.num_workers, drop_last=True, shuffle=True)

        val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
            num_workers=config.num_workers, drop_last=True, shuffle=False)

        # model defination
        net = MA_AGIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
            patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
            depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

        logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

        net = net.cuda()

        # loss function
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

        # train & validation
        losses, scores = [], []
        best_srocc = 0
        best_plcc = 0
        main_score = 0
        for epoch in range(0, config.n_epoch):
            start_time = time.time()
            logging.info('Running training epoch {}'.format(epoch + 1))
            loss_val, rho_s, rho_p = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)

            if (epoch + 1) % config.val_freq == 0:
                logging.info('Starting eval...')
                logging.info('Running testing in epoch {}'.format(epoch + 1))
                loss, rho_s, rho_p = eval_epoch(config, epoch, net, criterion, val_loader)
                logging.info('Eval done...')

                if rho_s + rho_p > main_score:
                    main_score = rho_s + rho_p
                    best_srocc = rho_s
                    best_plcc = rho_p

                    logging.info('======================================================================================')
                    logging.info('============================== best main score is {} ================================='.format(main_score))
                    logging.info('======================================================================================')

                    # save weights
                    os.makedirs("checkpt", exist_ok=True)
                    model_save_path = "checkpt/{}.pt".format(config.dataset_name)
                    torch.save(net.state_dict(), model_save_path)
                    logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))
            
            logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))
        
        #test
        test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
            num_workers=config.num_workers, drop_last=True, shuffle=True)
        
        # model defination
        net = MA_AGIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
            patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
            depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

        logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

        ckpt_path = "checkpt/{}.pt".format(config.dataset_name)
        net.load_state_dict(torch.load(ckpt_path))
        net = net.cuda()
        criterion = torch.nn.MSELoss()
        output_csv =  None
        srcc_all[i], plcc_all[i], krcc_all[i], rmse_all[i] = test_epoch(config, 1, net, criterion, test_loader, output_csv=output_csv)

        print('Testing SRCC %4.4f,\tPLCC %4.4f,\tKRCC %4.4f,\tRMSE %4.4f' % (srcc_all[i], plcc_all[i], krcc_all[i], rmse_all[i]))

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)
    krcc_med = np.median(krcc_all)
    rmse_med = np.median(rmse_all)

    srcc_std = np.std(srcc_all)
    plcc_std = np.std(plcc_all)
    krcc_std = np.std(krcc_all)
    rmse_std = np.std(rmse_all)


    print('training median SRCC %4.4f,\tmedian PLCC %4.4f,\tmedian KRCC %4.4f,\tmedian RMSE %4.4f' % (srcc_med, plcc_med,krcc_med, rmse_med))
    print('training std SRCC %4.4f,\tstd PLCC %4.4f,\tstd KRCC %4.4f,\tstd RMSE %4.4f' % (srcc_std, plcc_std,krcc_std, rmse_std))

    #log out to {dataset_name}.txt
    log_file = '{}.txt'.format(config.dataset_name)
    with open(log_file, 'a') as f:
        f.write('training median SRCC %4.4f,\tmedian PLCC %4.4f,\tmedian KRCC %4.4f,\tmedian RMSE %4.4f\n' % (srcc_med, plcc_med,krcc_med, rmse_med))
        f.write('training std SRCC %4.4f,\tstd PLCC %4.4f,\tstd KRCC %4.4f,\tstd RMSE %4.4f\n' % (srcc_std, plcc_std,krcc_std, rmse_std))