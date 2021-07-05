import sys
import torch
import copy
import numpy as np
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from constants.enum_keys import HG
# from models.hands_recognition_model import HandsRecognitionModel
from models.static_recognition_model import StaticRecognitionModel
from hgdataset.s3_handcraft import HgdHandcraft
from pred.evaluation import EditDistance
from constants import settings
from utils.wswa import wswaUpdate
from visdom import Visdom
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000


class Trainer:
    def __init__(self, is_unittest=False):
        self.is_unittest = is_unittest
        self.batch_size = 2
        self.clip_len = 15 * 30
        # self.clip_len = 15 * 90
        hgd_train = HgdHandcraft('static', Path.home() / 'MeetingHands', True, (512, 512), clip_len=self.clip_len)
        hgd_val = HgdHandcraft('static', Path.home() / 'MeetingHands', False, (512, 512), clip_len=self.clip_len)
        self.ed = EditDistance()
        self.train_loader = DataLoader(hgd_train, batch_size=self.batch_size, shuffle=True, num_workers=settings.num_workers)
        self.val_loader = DataLoader(hgd_val, batch_size=self.batch_size, shuffle=True, num_workers=settings.num_workers)
        hgd = HgdHandcraft('static', Path.home() / 'MeetingHands', True, (512, 512), clip_len=self.clip_len)
        self.data_loader = DataLoader(hgd, batch_size=self.batch_size, shuffle=False, num_workers=settings.num_workers)
        # self.model = HandsRecognitionModel(batch=self.batch_size)
        self.model = StaticRecognitionModel()
        self.model.train()
        self.model_folder = Path.cwd() / 'checkpoints'
        self.loss_his_train = []
        self.epochs = 1000
        self.lr = 0.008
        self.best_acc = 0.2
        self.loss = CrossEntropyLoss()
        # self.opt = optim.Adam(self.model.parameters(), self.lr)
        self.opt = optim.SGD(self.model.parameters(), momentum=0.9, lr=0.1, weight_decay=5e-4)
        # self.scheduler = CosineAnnealingLR(self.opt, T_max=self.epochs, eta_min=1e-8)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vis = Visdom()
 
    def adjust_learning_rate(self, optimizer, epoch):
        '''
        epoch:当前epoch
        fin_epoch：总共要训练的epoch数
        ini_lr:初始学习率
        lr:需要优化的学习率(optimizer中的学习率)
        '''
        lr = self.ADamLR(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def ADamLR(self, epoch, B=1000):
        a1 = self.lr
        C = 300
        rest = epoch % B
        if rest < 0.5 * C:
            return a1
        elif rest < 0.9 * C:
            return (0.9 * C - rest) / (0.9 * C - 0.5 * C) * (a1 - 0.01 * a1)
        else:
            return 0.01 * a1

    def set_train(self):
        """Convert models to training mode
        """
        self.model.train()

    def set_eval(self):
        """Convert models to testing/evaluation mode
        """
        self.model.eval()

    def train(self):
        self.epoch = 0
        for epoch in range(self.epochs):
            loss_trained = self.run_epoch(epoch)
            acc_valed = self.val(self.model)
            self.loss_his_train.append(loss_trained)
            print("Epoch:{}".format(epoch))
            print("train loss: {}".format(loss_trained))
            print("val accuracy: {}".format(acc_valed))
            model_path = self.model_folder / Path('static_model_epoch_' + str(epoch) + '_valacc_' + str(round(acc_valed, 2)) + '.pt')
            WSWA2_test_ap = 0
            WSWA3_test_ap = 0
            WSWA4_test_ap = 0
            WSWA5_test_ap = 0
            # wnw2 = None
            # wnw3 = None
            # wnw4 = None
            # wnw5 = None
            if epoch>=1000:
                if epoch == 1000:
                    # wnw = copy.deepcopy(self.model)
                    wnw2 = copy.deepcopy(self.model)
                    wnw3 = copy.deepcopy(self.model)
                    wnw4 = copy.deepcopy(self.model)
                    wnw5 = copy.deepcopy(self.model) 
                    # Sum_an_nw = acc_valed
                    Sum_an_nw2 = acc_valed
                    Sum_an_nw3 = acc_valed
                    Sum_an_nw4 = acc_valed
                    Sum_an_nw5 = acc_valed
                else:    
                    # wnw, Sum_an_nw = wswaUpdate(self.model, Sum_an_nw, acc_valed, wnw, w=0)
                    wnw2, Sum_an_nw2 = wswaUpdate(self.model, Sum_an_nw2, acc_valed, wnw2, w=0.3)
                    wnw3, Sum_an_nw3 = wswaUpdate(self.model, Sum_an_nw3, acc_valed, wnw3, w=0.4)
                    wnw4, Sum_an_nw4 = wswaUpdate(self.model, Sum_an_nw4, acc_valed, wnw4, w=0.5)
                    wnw5, Sum_an_nw5 = wswaUpdate(self.model, Sum_an_nw5, acc_valed, wnw5, w=0.6)

                # WSWA_test_ap = self.val(wnw)
                WSWA2_test_ap = self.val(wnw2)
                WSWA3_test_ap = self.val(wnw3)
                WSWA4_test_ap = self.val(wnw4)
                WSWA5_test_ap = self.val(wnw5)
                if WSWA2_test_ap > self.best_acc:
                    model_path_swa2 = self.model_folder / Path('lstm_swa2_model_epoch_' + str(epoch) + '_valacc_' + str(round(WSWA2_test_ap, 2)) + '.pt')
                    self.best_acc = WSWA2_test_ap
                    torch.save(wnw2.state_dict(), model_path_swa2)
                if WSWA3_test_ap > self.best_acc:
                    model_path_swa3 = self.model_folder / Path('lstm_swa3_model_epoch_' + str(epoch) + '_valacc_' + str(round(WSWA3_test_ap, 2)) + '.pt')
                    self.best_acc = WSWA3_test_ap
                    torch.save(wnw3.state_dict(), model_path_swa3)
                if WSWA4_test_ap > self.best_acc:
                    model_path_swa4 = self.model_folder / Path('lstm_swa4_model_epoch_' + str(epoch) + '_valacc_' + str(round(WSWA4_test_ap, 2)) + '.pt')
                    self.best_acc = model_path_swa4
                    torch.save(wnw4.state_dict(), model_path_swa4)
                if WSWA5_test_ap > self.best_acc:
                    model_path_swa5 = self.model_folder / Path('lstm_swa5_model_epoch_' + str(epoch) + '_valacc_' + str(round(WSWA5_test_ap, 2)) + '.pt')
                    self.best_acc = WSWA5_test_ap
                    torch.save(wnw5.state_dict(), model_path_swa5)
            name = ['train_loss', 'val_accuracy','wnw2_val','wnw3_val', 'wnw4_val', 'wnw5_val']
            self.vis.line(np.column_stack(([loss_trained.item()], [acc_valed], [WSWA2_test_ap], [WSWA3_test_ap], [WSWA4_test_ap], [WSWA5_test_ap])), [epoch], win='loss', update='append', opts=dict(title='losses', legend=name))
            if acc_valed < self.best_acc:
                self.best_acc = acc_valed
                torch.save(self.model.state_dict(), model_path)
        self.model.save_ckpt()

    def run_epoch(self, epoch):
        loss_train = torch.tensor(.0).to(self.device)
        train_count = 0
        self.set_train()
        for ges_data in self.train_loader:
            features = torch.cat((ges_data[HG.BONE_LENGTH], ges_data[HG.BONE_ANGLE_COS],
                                    ges_data[HG.BONE_ANGLE_SIN], ges_data[HG.BONE_DEPTH]), dim=2)
            features = features.permute(1, 0, 2)
            features = features.to(self.model.device, dtype=torch.float32)
            # h0, c0 = self.model.h0(), self.model.c0()
            # _, h, c, class_out = self.model(features, h0, c0)
            class_out = self.model(features)
            # class_out = class_out.squeeze(dim=1)
            target = ges_data[HG.GESTURE_LABEL]
            target = target.to(self.model.device, dtype=torch.long)
            target = target.permute(1, 0)
            target = target.reshape((-1))
            loss_step = self.loss(class_out, target)
            self.opt.zero_grad()
            loss_step.backward()
            self.opt.step()
            loss_train += loss_step
            train_count += 1
            # self.scheduler.step()
            self.adjust_learning_rate(self.opt, epoch)
        loss_trained = loss_train.cpu().detach().numpy() / train_count
        return loss_trained
    
    def val(self, model):
        loss_val = torch.tensor(.0).to(self.device)
        sumDistence = [0, 0, 0, 0]
        val_count = 0
        self.set_eval()
        for ges_data in self.val_loader:
            features = torch.cat((ges_data[HG.BONE_LENGTH], ges_data[HG.BONE_ANGLE_COS],
                                    ges_data[HG.BONE_ANGLE_SIN], ges_data[HG.BONE_DEPTH]), dim=2)
            features = features.permute(1, 0, 2)
            features = features.to(model.device, dtype=torch.float32)
            class_out = model(features)
            class_out = class_out.squeeze(dim=1)
            target = ges_data[HG.GESTURE_LABEL]
            target = target.to(model.device, dtype=torch.long)
            target = target.permute(1, 0)
            target = target.reshape((-1))
            loss_step = self.loss(class_out, target)
            loss_val += loss_step
            val_count += 1
        val_loss = loss_val.cpu().detach().numpy() / val_count
        return val_loss

