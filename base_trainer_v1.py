from dataclasses import dataclass
import os
import logging
from random import shuffle

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.utils import accuracy, AverageMeter

from utils.utils import save_model, write_log
from utils.lr_scheduler import inv_lr_scheduler
from datasets import *
from models import *
import tqdm
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torchvision.transforms as T
from torch.cuda import amp
import torch.distributed as dist
import logging


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        logging.info(f'--> trainer: {self.__class__.__name__}')

        self.setup()
        self.build_datasets()
        self.build_models()
        self.resume_from_ckpt()

    def setup(self):
        self.start_ite = 0
        self.ite = 0
        self.best_acc = 0.
        self.tb_writer = SummaryWriter(self.cfg.TRAIN.OUTPUT_TB)

    def build_datasets(self):
        logging.info(f'--> building dataset from: {self.cfg.DATASET.NAME}')
        self.dataset_loaders = {}

        # dataset loaders
        if self.cfg.DATASET.NAME == 'office_home':
            dataset = OfficeHome
        elif self.cfg.DATASET.NAME == 'domainnet':
            dataset = DomainNet
        else:
            raise ValueError(f'Dataset {self.cfg.DATASET.NAME} not found')
        self.source_dataset = dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.SOURCE, status='val')
        self.target_dataset = dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.TARGET, status='val')
        self.dataset_loaders['source_train'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.SOURCE, status='train'),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_SOURCE,
            shuffle=True,
            num_workers=self.cfg.WORKERS,
            drop_last=True
        )
        self.dataset_loaders['source_train_p'] = DataLoader(
            self.source_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE_SOURCE,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )
        self.dataset_loaders['source_test'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.SOURCE, status='val', trim=self.cfg.DATASET.TRIM),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )
        self.dataset_loaders['target_train'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.TARGET, status='train'),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TARGET,
            shuffle=True,
            num_workers=self.cfg.WORKERS,
            drop_last=True
        )
        self.dataset_loaders['target_train_p'] = DataLoader(
            self.target_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TARGET,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )
        self.dataset_loaders['target_test'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.TARGET, status='test'),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )
        self.len_src = len(self.dataset_loaders['source_train'])
        self.len_tar = len(self.dataset_loaders['target_train'])
        logging.info(f'    source {self.cfg.DATASET.SOURCE}: {self.len_src}'
                     f'/{len(self.dataset_loaders["source_test"])}')
        logging.info(f'    target {self.cfg.DATASET.TARGET}: {self.len_tar}'
                     f'/{len(self.dataset_loaders["target_test"])}')

    def build_models(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        self.base_net = self.build_base_models(mlp=True)
        self.registed_models = {'base_net': self.base_net}
        # print('------params', self.base_net)
        parameter_list = self.base_net.get_parameters()
        self.model_parameters()
        self.build_optim(parameter_list, parameter_list_store)

    def build_base_models(self, mlp=None):
        # print('==============num_classes', self.cfg.DATASET.NUM_CLASSES)
        basenet_name = self.cfg.MODEL.BASENET
        kwargs = {
            'pretrained': self.cfg.MODEL.PRETRAIN,
            'num_classes': self.cfg.DATASET.NUM_CLASSES,
            'mlp': False
        }

        basenet = eval(basenet_name)(**kwargs).cuda()

        return basenet

    def model_parameters(self):
        for k, v in self.registed_models.items():
            logging.info(f'    {k} paras: '
                         f'{(sum(p.numel() for p in v.parameters()) / 1e6):.2f}M')

    def build_optim(self, parameter_list: list, parameter_list_store: list):
        self.optimizer = optim.SGD(
            parameter_list,
            lr=self.cfg.TRAIN.LR,
            momentum=self.cfg.OPTIM.MOMENTUM,
            weight_decay=self.cfg.OPTIM.WEIGHT_DECAY,
            nesterov=True
        )

        self.optimizer_s = None
        self.lr_scheduler = inv_lr_scheduler

    def resume_from_ckpt(self):
        last_ckpt = os.path.join(self.cfg.TRAIN.OUTPUT_CKPT, 'models-last.pt')
        if os.path.exists(last_ckpt):
            ckpt = torch.load(last_ckpt)
            for k, v in self.registed_models.items():
                v.load_state_dict(ckpt[k])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_ite = ckpt['ite']
            self.best_acc = ckpt['best_acc']
            logging.info(f'> loading ckpt from {last_ckpt} | ite: {self.start_ite} | best_acc: {self.best_acc:.3f}')
        else:
            logging.info('--> training from scratch')


    # train only one time source data.

    def train(self):
        logger = logging.getLogger("reid_baseline.train")
        logger.info('start training')
        # start training
        for _, v in self.registed_models.items():
            v.train()
        # step one
    
        for self.ite in range(self.start_ite, self.cfg.TRAIN.TTL_ITE):
            # test
            # print('base_model', self.base_net)
            if (self.ite) % self.cfg.TRAIN.TEST_FREQ == self.cfg.TRAIN.TEST_FREQ - 1 and self.ite != self.start_ite:
                self.base_net.eval()
                self.test()
                self.base_net.train()

            self.current_lr = self.lr_scheduler(
                self.optimizer,
                ite_rate=self.ite,
                lr=self.cfg.TRAIN.LR,
            )

            # dataloader
            if self.ite % self.len_src == 0 or self.ite == self.start_ite:
                iter_src = iter(self.dataset_loaders['source_train'])
            if self.ite % self.len_tar == 0 or self.ite == self.start_ite:
                iter_tar = iter(self.dataset_loaders['target_train'])

            # forward one iteration
            data_src = iter_src.__next__()
            data_tar = iter_tar.__next__()
         
            self.one_step(data_src, data_tar)
            if self.ite % self.cfg.TRAIN.SAVE_FREQ == 0 and self.ite != 0:
                self.save_model(is_best=False, snap=True)

    
        ###################################stage 2 load_weights###################
        if self.ite < self.start_ite:
            start_epoch_2 = self.start_ite
        else: 
            start_epoch_2 = self.cfg.TRAIN.TTL_ITE
        self.test_freq_2 = 500
        #########################################################################

        # self.pseudo_num = 0
        # self.test_freq_2 = 100
        for self.ite in range(self.start_ite, self.cfg.TRAIN.TTL_ITE+6000):
            # test
            if self.ite == self.start_ite:
                start_pseduo = True
            if ((self.ite-self.cfg.TRAIN.TTL_ITE) % self.test_freq_2 == self.test_freq_2 - 1 and self.ite != self.start_ite) or start_pseduo==True:
                self.base_net.eval()
                self.test()
                if self.pseudo_num % 5==0:
                    start_pseduo = False
                    img_num1 = len(self.source_dataset.data)
                    img_num2 = len(self.target_dataset.data)
                    label_memory1 = torch.zeros((img_num1),dtype=torch.long)
                    label_memory2 = torch.zeros((img_num2),dtype=torch.long)

                    feat_memory1 = torch.zeros((img_num1,2048),dtype=torch.float32)  #2048 can be adjusted based on your model output before classifer
                    feat_memory2 = torch.zeros((img_num2,2048),dtype=torch.float32)
                    feat_memory1, feat_memory2, label_memory1, label_memory2 = self.update_feat(self.base_net, self.dataset_loaders['source_train_p'], self.dataset_loaders['target_train_p'], device='cuda',feat_memory1=feat_memory1,feat_memory2=feat_memory2, label_memory1=label_memory1,label_memory2=label_memory2)
                    dynamic_top = 1
                    print('source and target topk==',dynamic_top)
                    target_label, knnidx, knnidx_topk, target_knnidx = self.compute_knn_idx(logger, self.base_net, self.dataset_loaders['source_train_p'], self.dataset_loaders['target_train_p'], feat_memory1, feat_memory2, label_memory1, label_memory2, img_num1, img_num2, topk=dynamic_top, reliable_threshold=0.0)

                    self.generate_new_dataset(logger, label_memory2, self.source_dataset.data, self.target_dataset.data, knnidx, target_knnidx, target_label, label_memory1, img_num1, img_num2, with_pseudo_label_filter=True)

                    iter_pseduo = iter(self.dataset_loaders['train_st'])
                    ite_num_pseduo = 0
                    self.len_stp = len(self.dataset_loaders['train_st'])
                self.pseudo_num +=1
                self.base_net.train()

            self.current_lr = self.lr_scheduler(
                self.optimizer,
                # ite_rate=(self.ite-self.cfg.TRAIN.TTL_ITE)*6,
                ite_rate = self.ite,
                lr=self.cfg.TRAIN.LR,
            )

            # dataloader
            if self.ite % self.len_src == 0 or self.ite == self.start_ite:
                iter_src = iter(self.dataset_loaders['source_train'])
            if self.ite % self.len_tar == 0 or self.ite == self.start_ite:
                iter_tar = iter(self.dataset_loaders['target_train'])
            if ite_num_pseduo % self.len_stp == 0:
                iter_pseduo = iter(self.dataset_loaders['train_st'])
                ite_num_pseduo = 0

            # forward one iteration
            data_src = iter_src.__next__()
            data_tar = iter_tar.__next__()
          
            data_stp = iter_pseduo.__next__()
          
            self.one_step_pseudo(data_src, data_stp, data_tar)
           
            if self.ite % self.cfg.TRAIN.SAVE_FREQ == 0 and self.ite != 0:
                self.save_model(is_best=False, snap=True)
        
            ite_num_pseduo+=1

    
    
    def one_step(self, data_src, data_tar):
        inputs_src, labels_src = data_src['image'].cuda(), data_src['label'].cuda()
        inputs_tar, labels_tar = data_tar['image'].cuda(), data_tar['label'].cuda()

        inputs_all = torch.cat((inputs_src, inputs_tar))
        _, logits_all, outputs_all_dc = self.base_net(inputs_all)
  
        logits_tar = logits_all[inputs_src.size(0):inputs_src.size(0)+inputs_tar.size(0)]

        classification_loss = nn.CrossEntropyLoss()(logits_all.narrow(0, 0, labels_src.size(0)), labels_src)
        domain_labels = torch.cat((torch.ones(inputs_src.shape[0], device=inputs_src.device, dtype=torch.float),
                 torch.zeros(inputs_tar.shape[0], device=inputs_tar.device, dtype=torch.float)),0)

        domain_loss = nn.BCELoss()(F.sigmoid(outputs_all_dc.narrow(0, 0, inputs_all.size(0))).squeeze(), domain_labels) * 2
        loss_alg = domain_loss
        ent_tar = entropy_func(nn.Softmax(dim=1)(logits_tar.data)).mean()

        # classificaiton
        loss_cls_src = classification_loss
        loss_cls_tar = F.cross_entropy(logits_tar.data, labels_tar)

       
        # domain alignment
        loss_ttl = loss_cls_src + loss_alg
       


        # update
        self.step(loss_ttl)

        # display
        if self.ite % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_cls_tar: {loss_cls_tar.item():.3f}',
                f'l_alg: {loss_alg.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'ent_tar: {ent_tar.item():.3f}',
                f'best_acc: {self.best_acc:.3f}',
            ])
            # tensorboard
            self.update_tb({
                'l_cls_src': loss_cls_src.item(),
                'l_cls_tar': loss_cls_tar.item(),
                'l_alg': loss_alg.item(),
                'l_ttl': loss_ttl.item(),
                'ent_tar': ent_tar.item(),
            })

    def one_step_pseudo(self, data_src, data_stp, data_tar):
        inputs_src, labels_src = data_src['image'].cuda(), data_src['label'].cuda()
        inputs_stp, labels_stp = data_stp['image'].cuda(), data_stp['label'].cuda()
        inputs_tar, labels_tar = data_tar['image'].cuda(), data_tar['label'].cuda()
        

        inputs_all = torch.cat((inputs_src, inputs_stp, inputs_tar))
        _, logits_all, outputs_all_dc = self.base_net(inputs_all, perturbation=True)

        logits_tar = logits_all[inputs_src.size(0)+inputs_stp.size(0):inputs_src.size(0)+inputs_stp.size(0)+inputs_tar.size(0)]

        classification_loss = nn.CrossEntropyLoss()(logits_all.narrow(0, 0, labels_src.size(0)+labels_stp.size(0)), torch.cat([labels_src, labels_stp], 0))
        domain_labels = torch.cat((torch.ones(inputs_src.shape[0], device=inputs_src.device, dtype=torch.float),
                 torch.zeros(inputs_tar.shape[0], device=inputs_tar.device, dtype=torch.float)),0)

        domain_loss = nn.BCELoss()(F.sigmoid(torch.cat([outputs_all_dc.narrow(0, 0, inputs_src.size(0)), outputs_all_dc.narrow(0, labels_src.size(0)+labels_stp.size(0), inputs_all.size(0)], 0))).squeeze(), domain_labels) * 2
        loss_alg = domain_loss

        # outputs_all_src = self.base_net(inputs_src)
        # outputs_all_tar = self.base_net(inputs_tar)

        ent_tar = entropy_func(nn.Softmax(dim=1)(logits_tar.data)).mean()

        # classificaiton
        loss_cls_src = classification_loss
        loss_cls_tar = F.cross_entropy(logits_tar.data, labels_tar)

       
        # domain alignment
        loss_ttl = loss_cls_src + loss_alg


        # update
        self.step(loss_ttl)

        # display
        if self.ite % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_cls_tar: {loss_cls_tar.item():.3f}',
                f'l_alg: {loss_alg.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'ent_tar: {ent_tar.item():.3f}',
                f'best_acc: {self.best_acc:.3f}',
            ])
            # tensorboard
            self.update_tb({
                'l_cls_src': loss_cls_src.item(),
                'l_cls_tar': loss_cls_tar.item(),
                'l_alg': loss_alg.item(),
                'l_ttl': loss_ttl.item(),
                'ent_tar': ent_tar.item(),
            })

    def display(self, data: list):
        log_str = f'I:  {self.ite}/{self.cfg.TRAIN.TTL_ITE} | lr: {self.current_lr:.5f} '
        # update
        for _str in data:
            log_str += '| {} '.format(_str)
        logging.info(log_str)

    def update_tb(self, data: dict):
        for k, v in data.items():
            self.tb_writer.add_scalar(k, v, self.ite)

    def step(self, loss_ttl):
        self.optimizer.zero_grad()
        loss_ttl.backward()
        self.optimizer.step()

    def test(self):
        logging.info('--> testing on source_test')
        src_acc = self.test_func(self.dataset_loaders['source_test'])
        logging.info('--> testing on target_test')
        tar_acc = self.test_func(self.dataset_loaders['target_test'])
    
        is_best = False
        if tar_acc > self.best_acc:
            self.best_acc = tar_acc
            is_best = True

        # display
        log_str = f'I:  {self.ite}/{self.cfg.TRAIN.TTL_ITE} | src_acc: {src_acc:.3f} | tar_acc: {tar_acc:.3f} | ' \
                  f'best_acc: {self.best_acc:.3f}'
        logging.info(log_str)

        # save results
        log_dict = {
            'I': self.ite,
            'src_acc': src_acc,
            'tar_acc': tar_acc,
            'best_acc': self.best_acc
        }
        write_log(self.cfg.TRAIN.OUTPUT_RESFILE, log_dict)

        # tensorboard
        self.tb_writer.add_scalar('tar_acc', tar_acc, self.ite)
        self.tb_writer.add_scalar('src_acc', src_acc, self.ite)

        self.save_model(is_best=is_best)


    def test_func(self, loader, model):
        with torch.no_grad():
            iter_test = iter(loader)
            print_freq = max(len(loader) // 5, self.cfg.TRAIN.PRINT_FREQ)
            accs = AverageMeter()
            for i in range(len(loader)):
                if i % print_freq == print_freq - 1:
                    logging.info('    I:  {}/{} | acc: {:.3f}'.format(i, len(loader), accs.avg))
                data = iter_test.__next__()
                inputs, labels = data['image'].cuda(), data['label'].cuda()
                _, outputs = model(inputs)  # [f, y, ...]
                # outputs = outputs_all[3]

                acc = accuracy(outputs, labels)[0]
                accs.update(acc.item(), labels.size(0))

        return accs.avg

    def save_model(self, is_best=False, snap=False):
        data_dict = {
            'optimizer': self.optimizer.state_dict(),
            'ite': self.ite,
            'best_acc': self.best_acc
        }
        for k, v in self.registed_models.items():
            data_dict.update({k: v.state_dict()})
        save_model(self.cfg.TRAIN.OUTPUT_CKPT, data_dict=data_dict, ite=self.ite, is_best=is_best, snap=snap)

    def obtain_label(self, logger, val_loader, model, distance='cosine', threshold=0):
        device = "cuda"
        start_test = True
        print('obtain label')
        model.eval()
        for n_iter, data in enumerate(val_loader):
            with torch.no_grad():
                image, label = data['image'].cuda(), data['label']
                outputs_all = model(image)
                feas, outputs = outputs_all[0], outputs_all[1]
        
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = label.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, label.float()), 0)

        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        if distance == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        ### all_fea: extractor feature [bs,N]

        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        ### aff: softmax output [bs,c]

        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        #print('-----------threshod', threshold, cls_count)
        labelset = np.where(cls_count > threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        log_str = 'Fisrt Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

        logger.info(log_str)
        
        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc[labelset], distance)
            pred_label = dd.argmin(axis=1)
            pred_label = labelset[pred_label]
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        log_str = 'Second Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        logger.info(log_str)
        #print('----------num_pred_label', len(pred_label))

        return pred_label.astype('int')

    def update_feat(self, model, train_source_loader,train_target_loader, device,feat_memory1,feat_memory2, label_memory1,label_memory2):
        model.eval()
        for n_iter, data in enumerate(train_source_loader):
            with torch.no_grad():
                img, label, idx = data['image'].cuda(), data['label'], data['index']
                feats = model(img)
                feat = feats[0]/(torch.norm(feats[0],2,1,True)+1e-8)
                feat_memory1[idx] = feat.detach().cpu()
                label_memory1[idx] = label

        for n_iter, data in enumerate(train_target_loader):
            with torch.no_grad():
                img, label, idx = data['image'].cuda(), data['label'], data['index']
                feats = model(img)
                feat = feats[0]/(torch.norm(feats[0],2,1,True)+1e-8)
                feat_memory2[idx] = feat.detach().cpu()
                label_memory2[idx] = label


        return feat_memory1, feat_memory2, label_memory1, label_memory2
    

    def compute_knn_idx(self, logger, model, train_source_loader, train_target_loader, feat_memory1, feat_memory2, label_memory1, label_memory2, img_num1, img_num2, target_sample_num=2, topk=1, reliable_threshold=0.0):
        #assert((torch.sum(feat_memory2,axis=1)!=0).all())
        simmat = torch.matmul(feat_memory1,feat_memory2.T)
        _, knnidx = torch.max(simmat,1)

        if topk == 1:
            target_knnsim, target_knnidx = torch.max(simmat, 0)
        else:
            target_knnsim, target_knnidx = torch.topk(simmat,dim=0,k=topk)
            target_knnsim, target_knnidx = target_knnsim[topk-1, :], target_knnidx[topk-1, :]

        _, knnidx_topk = torch.topk(simmat,k=target_sample_num,dim=1)
        del simmat
        self.count_target_usage(logger, knnidx, label_memory1, label_memory2, img_num1, img_num2)

        target_label = self.obtain_label(logger, train_target_loader, model)
        print('-----target_label', len(target_label))
        target_label = torch.from_numpy(target_label).cuda()

        return target_label, knnidx, knnidx_topk, target_knnidx

    def count_target_usage(self, logger, idxs, label_memory1, label_memory2, img_num1, img_num2, source_idxs=None):
        unique_knnidx = torch.unique(idxs)
        logger.info('target number usage: {}'.format(len(unique_knnidx)/img_num2))
        if source_idxs is not None:
            source_unique_knnidx = torch.unique(source_idxs)
            logger.info('source number usage: {}'.format(len(source_unique_knnidx)/img_num1))
        else:
            logger.info('source number usage: 100%')

        per_class_num = torch.bincount(label_memory2)
        per_class_select_num = torch.bincount(label_memory2[unique_knnidx])
        logger.info('target each class usage: {} '.format(per_class_select_num/per_class_num[:len(per_class_select_num)]))
        if len(per_class_num) != len(per_class_select_num):
            logger.info('target last {} class usage is 0%'.format(len(per_class_num) - len(per_class_select_num)))
        
        target_labels = label_memory2[idxs]
        if source_idxs is not None: # sample should filter
            source_labels = label_memory1[source_idxs]
        else:
            source_labels = label_memory1

        logger.info('match right rate: {}'.format((target_labels==source_labels).int().sum()/len(target_labels)))
        matrix = confusion_matrix(target_labels, source_labels)
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aa = [str(np.round(i, 2)) for i in acc]
        logger.info('each target class match right rate: {}'.format(aa))

    
    def generate_new_dataset(self, logger, label_memory2, s_dataset, t_dataset, knnidx, target_knnidx, target_pseudo_label, label_memory1, img_num1, img_num2, with_pseudo_label_filter=True):
    # generate new dataset
        train_set = []
        train_set_single = [] # no pair training
        new_target_knnidx = []
        new_targetidx = []
        
        source_dataset = s_dataset
        target_dataset = t_dataset
        #print('---------tttttttt', target_dataset)
        # combine_target_sample:
        for idx, data in enumerate(target_dataset):
            t_img_path, t_label = data
            curidx = target_knnidx[idx]
            if curidx<0: continue
            source_data = source_dataset[curidx]
            s_img_path, label  = source_data
            mask = label == target_pseudo_label[idx]
            if (with_pseudo_label_filter and mask) or not with_pseudo_label_filter:
                new_targetidx.append(idx)
                new_target_knnidx.append(curidx)
                train_set_single.append((t_img_path, target_pseudo_label[idx].item(), idx))
        logger.info('target match accuracy') 
        self.count_target_usage(logger, torch.tensor(new_targetidx), label_memory1, label_memory2, img_num1, img_num2, source_idxs=torch.tensor(new_target_knnidx))
        print('---------target_-----len_s', len(train_set))
        # combine_source_sample:
        new_source_knnidx = []
        new_source_idx = []
        for idx, data in enumerate(source_dataset):
            s_img_path, label = data
            curidx = knnidx[idx]
            if curidx<0:continue
            target_data = target_dataset[curidx]
            t_img_path, t_label  = target_data
            mask = target_pseudo_label[curidx] == label
            if (with_pseudo_label_filter and mask) or not with_pseudo_label_filter:
                new_source_idx.append(idx)
                new_source_knnidx.append(curidx)
                train_set_single.append((t_img_path, target_pseudo_label[curidx].item(), curidx.item()))
        logger.info('source match accuracy') 
        self.count_target_usage(logger, torch.tensor(new_source_knnidx), label_memory1, label_memory2, img_num1, img_num2, source_idxs=torch.tensor(new_source_idx))
        
        new_target_knnidx = new_target_knnidx + new_source_idx
        new_targetidx = new_targetidx + new_source_knnidx 

        self.count_target_usage(logger, torch.tensor(new_targetidx), label_memory1, label_memory2, img_num1, img_num2, source_idxs=torch.tensor(new_target_knnidx))
        shuffle(train_set_single)
        new_dataset = CommonDatasetP(cfg=self.cfg, data=train_set_single, is_train=True)

        self.dataset_loaders['train_st'] = DataLoader(
            new_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TARGET,
            shuffle=True,
            num_workers=self.cfg.WORKERS,
            drop_last=True
        )
        


