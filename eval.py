import datasets
from GaMPF import GaMPF
import os
import cv2
import torch
import numpy as np
from os.path import join as pjoin
import torch.nn.functional as F
import argparse

class Evaluate:
    def __init__(self):
        self.args = None
        self.set = None

    def store_imgs_and_cal_matrics(self, t0, t1, mask_gt, mask_pred):
        w, h = self.w_r, self.h_r
        img_save = np.zeros((w * 2, h * 2, 3), dtype=np.uint8)
        img_save[0:w, 0:h, :] = np.transpose(t0.numpy(), (1, 2, 0)).astype(np.uint8)
        img_save[0:w, h:h * 2, :] = np.transpose(t1.numpy(), (1, 2, 0)).astype(np.uint8)
        img_save[w:w * 2, 0:h, :] = cv2.cvtColor(mask_gt.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_save[w:w * 2, h:h * 2, :] = cv2.cvtColor(mask_pred.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        fn_save = self.fn_img
        if not os.path.exists(self.dir_img):
            os.makedirs(self.dir_img)

        print('save...' + fn_save)
        cv2.imwrite(fn_save, img_save)

    def Init(self):
        self.resultdir = pjoin(self.args.resultdir, self.args.dataset)
        if not os.path.exists(self.resultdir):
            os.makedirs(self.resultdir)

    def run(self):
        print("load..." + self.fn_model)
        self.model = GaMPF(self.args.encoder_arch)
        self.model.load_state_dict(torch.load(self.fn_model))
        self.model = self.model.cuda()
        self.model.eval()

class evaluate_clcd(Evaluate):

    def __init__(self, arguments):
        super(evaluate_clcd, self).__init__()
        self.args = arguments

    def Init(self):
        super(evaluate_clcd,self).Init()
        self.ds = None
        self.index = 0
        self.dir_img = pjoin(self.resultdir)
        self.fn_model = pjoin(self.args.checkpointdir, 'xxxxxx.pth') #model_name

    def eval(self):
        input = torch.from_numpy(np.concatenate((self.t0,self.t1),axis=0)).contiguous()
        input = input.view(1,-1,self.w_r,self.h_r)
        input = input.cuda()
        output= self.model(input)

        input = input[0].cpu().data
        img_t0 = input[0:3,:,:]
        img_t1 = input[3:6,:,:]
        img_t0 = (img_t0+1)*128
        img_t1 = (img_t1+1)*128
        output = output[0].cpu().data
        mask_pred = np.where(F.softmax(output[0:2,:,:],dim=0)[0]>0.5, 0, 255)
        mask_gt = np.squeeze(np.where(self.mask==True,255,0),axis=0)
        self.store_imgs_and_cal_matrics(img_t0,img_t1,mask_gt,mask_pred)

    def run(self):
        super(evaluate_clcd, self).run()
        test_loader = datasets.clcd_eval(pjoin(self.args.datadir))

        for i in range(0, len(test_loader)):
            self.index = i
            self.fn_img = pjoin(self.dir_img, '{0}.png'.format(self.index))
            self.t0, self.t1, self.mask, self.w_ori, self.h_ori, self.w_r, self.h_r = test_loader[i]
            self.eval()


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Arguments for testing")
    parser.add_argument('--dataset', type=str, default='clcd')
    parser.add_argument('--datadir',required=True)
    parser.add_argument('--resultdir',required=True)
    parser.add_argument('--checkpointdir',required=True)
    parser.add_argument('--encoder-arch', type=str, default='resnet18')
    parser.add_argument('--store-imgs', action='store_true')

    eval = evaluate_clcd(parser.parse_args())
    eval.Init()
    eval.run()