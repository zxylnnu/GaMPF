import os
import torch
from GaMPF import GaMPF
import datasets
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join as pjoin
from torch.utils.data import DataLoader
import argparse

class criterion_CEloss(nn.Module):
    def __init__(self,weight=None):
        super(criterion_CEloss, self).__init__()
        self.loss = nn.NLLLoss(weight)
    def forward(self,output,target):
        return self.loss(F.log_softmax(output, dim=1), target)

class Train:

    def __init__(self):
        self.epoch = 0
        self.step = 0

    def train(self):

        weight = torch.ones(2)
        criterion = criterion_CEloss(weight.cuda())
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001,betas=(0.9,0.999))
        lambda_lr = lambda epoch:(float)(self.args.max_epochs*len(self.dataset_train_loader)-self.step)/(float)(self.args.max_epochs*len(self.dataset_train_loader))
        model_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda_lr)
        loss_item = []

        while self.epoch < self.args.max_epochs:
            for step,(inputs_train,mask_train) in enumerate(tqdm(self.dataset_train_loader)):
                self.model.train()
                inputs_train = inputs_train.cuda()
                mask_train = mask_train.cuda()

                output_train = self.model(inputs_train)
                optimizer.zero_grad()
                self.loss = criterion(output_train, mask_train[:,0])
                loss_item.append(self.loss)
                self.loss.backward()
                optimizer.step()
                self.step += 1

            print('Loss for Epoch {}:{:.06f}'.format(self.epoch, sum(loss_item)/len(self.dataset_train_loader)))
            loss_item.clear()
            model_lr_scheduler.step()
            self.epoch += 1
            if self.args.epoch_save>0 and self.epoch % self.args.epoch_save == 0:
                self.checkpoint()

    def checkpoint(self):
        filename = '{:08d}.pth'.format(self.step)
        cp_path = pjoin(self.checkpoint_save)
        if not os.path.exists(cp_path):
            os.makedirs(cp_path)
        torch.save(self.model.state_dict(),pjoin(cp_path,filename))
        print("saving...".format(self.step))

    def run(self):
        self.model = GaMPF(self.args.encoder_arch)
        self.model = self.model.cuda()
        self.train()

class train_clcd(Train):

    def __init__(self, arguments):
        super(train_clcd, self).__init__()
        self.args = arguments

    def Init(self):

        self.dataset_train_loader = DataLoader(datasets.clcd(pjoin(self.args.datadir, "train")),
                                          num_workers=self.args.num_workers, batch_size=self.args.batch_size,
                                          shuffle=True)
        self.dataset_val = datasets.clcd(pjoin(self.args.datadir, 'val'))
        self.checkpoint_save = pjoin(self.args.checkpointdir, 'clcd')
        if not os.path.exists(self.checkpoint_save):
            os.makedirs(self.checkpoint_save)


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument('--checkpointdir', required=True)
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--encoder-arch', type=str, default='resnet18')
    parser.add_argument('--max-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epoch-save', type=int, default=25)

    train = train_clcd(parser.parse_args())
    train.Init()
    train.run()









