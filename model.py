import collections

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection.image_list import ImageList
import torch.nn as nn
import torch
import torch.nn.functional as F
from backbone import ResNet, DenseNet 
from config import HyperParams
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from creatrpn import define_rpn
from torchvision.ops import RoIAlign

nums = 0


class TopkPool(nn.Module):
    def __init__(self):
        super(TopkPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.shape
        x = x.view(b, c, -1)
        topkv, _ = x.topk(5, dim=-1)
        return topkv.mean(dim=-1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FDM(nn.Module):
    def __init__(self):
        super(FDM, self).__init__()
        self.factor = round(1.0/(28*28), 3)

    def forward(self, fm1, fm2):
        b, c, w1, h1 = fm1.shape
        _, _, w2, h2 = fm2.shape
        fm1 = fm1.view(b, c, -1)    # B*C*S
        fm2 = fm2.view(b, c, -1)    # B*C*M

        fm1_t = fm1.permute(0, 2, 1)    # B*S*C

        # may not need to normalize
        fm1_t_norm = F.normalize(fm1_t, dim=-1)
        fm2_norm = F.normalize(fm2, dim=1)
        M = -1 * torch.bmm(fm1_t_norm, fm2_norm)    # B*S*M

        M_1 = F.softmax(M, dim=1)
        M_2 = F.softmax(M.permute(0, 2, 1), dim=1)
        new_fm2 = torch.bmm(fm1, M_1).view(b, c, w2, h2)
        new_fm1 = torch.bmm(fm2, M_2).view(b, c, w1, h1)

        return self.factor*new_fm1, self.factor * new_fm2


class FBSD(nn.Module):
    def __init__(self, class_num, arch='resnet50'):
        super(FBSD, self).__init__()
        feature_size = 512
        if arch == 'resnet50':
            self.features = ResNet(arch='resnet50')
            chans = [512, 1024, 2048]
        elif arch == 'resnet101':
            self.features = ResNet(arch='resnet101')
            chans = [512, 1024, 2048]
        elif arch == 'densenet161':
            self.features = DenseNet()
            chans = [768, 2112, 2208]

        self.pool = TopkPool()

        part_feature = 1024

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(part_feature* 3),
            nn.Linear(part_feature* 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.conv_block1 = nn.Sequential(
            BasicConv(chans[0], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(chans[1], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(chans[2], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.classifierrpn1 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.classifierrpn2 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.classifierrpn3 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.FPN = FeaturePyramidNetwork([1024,1024,1024],1024)


        #self.catrpn_conv = nn.Conv2d(5120, 1024, 1, 1, 0)
       # self.catrpn_conv = nn.Sequential(
        #    BasicConv(5120, 1024, kernel_size=1, stride=1, padding=0, relu=True),
        #    BasicConv(1024, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        #)
        self.roialign1 = RoIAlign([32, 32], 1.0, -1)
        self.roialign2 = RoIAlign([16, 16], 1.0, -1)
        self.roialign3 = RoIAlign([8, 8], 1.0, -1)
        self.inter = FDM()
        #self.rpn1 = define_rpn(128)
        #self.rpn2 = define_rpn(64)
        self.rpn = define_rpn(256)

    def forward(self, x):
        show = x.clone()
        fm1, fm2, fm3 = self.features(x)

        #########################################
        ##### cross-level attention #############
        #########################################

        att1 = self.conv_block1(fm1)
        att2 = self.conv_block2(fm2)
        att3 = self.conv_block3(fm3)

        new_d1_from2, new_d2_from1 = self.inter(att1, att2)  # 1 2
        new_d1_from3, new_d3_from1 = self.inter(att1, att3)  # 1 3
        new_d2_from3, new_d3_from2 = self.inter(att2, att3)  # 2 3

        gamma = HyperParams['gamma']
        att1 = att1 + gamma*(new_d1_from2 + new_d1_from3)
        att2 = att2 + gamma*(new_d2_from1 + new_d2_from3)
        att3 = att3 + gamma*(new_d3_from1 + new_d3_from2)
        #print(att1.size())(C , 1024, 32, 32)
        #print(att2.size())(C , 1024, 16, 16)
        #print(att3.size())(C , 1024, 8, 8)
        ####################
        d = collections.OrderedDict()
        d['0'] = att1
        d['1'] = att2
        d['2'] = att3
        features = self.FPN(d)

        ###################################################################

        #########rpnxian
        batchs = att1.size()[0]
        images = []
        for i in range(batchs):
            images.append([300, 300])
        images = ImageList(features['0'], images)
        
        tmp1 = collections.OrderedDict()
        tmp1['0'] = features['0']
        rpn1_out, _ = self.rpn(images, tmp1)
        #vis_box(show,rpn1_out)
        #print(len(rpn1_out))
        roi_in1 = []
        for i in rpn1_out:
            roi_in1.append(i[0:10,:])
        # show_box = []
        # for i in range(show.size()[0]):
        #     show_feature = show[i]
        #     show_box.append(rpn1_out[i][0:5, :])
        #     self.vis_box(show_feature, show_box)
        #     show_box.clear()
        roi1_out = self.single_cat(roi_in1, features['0'], self.roialign1)
        # roi1_out = self.roialign1(features['0'], list(roi_in1))
        # #print(roi1_out.size())
        # #print(roi1_out)1
        # #print(features['0'].size())
        features['0'] = features['0'] + roi1_out
        xl1 = self.pool(features['0'])
        xc1 = self.classifier1(xl1)

        ####################################################################

        ####################################################################
        tmp2 = collections.OrderedDict()
        tmp2['0'] = features['1']
        
        rpn2_out, _ = self.rpn(images, tmp2)
        roi_in2 = []
        #print(len(rpn2_out))
        for i in rpn2_out:
            roi_in2.append(i[0:10,:])
        ####visbox
        # show_box = []
        # for i in range(show.size()[0]):
        #     show_feature=show[i]
        #     show_box.append(rpn2_out[i][0:5,:])
        #     self.vis_box(show_feature,show_box)
        #     show_box.clear()
        ####4

        #roi2_out = self.roialign2(features['1'], list(roi_in2))
        #print(features['1'].size())
        roi2_out = self.single_cat(roi_in2, features['1'], self.roialign2)
        # #print(roi2_out.size())
        features['1'] = features['1']+ roi2_out
        xl2 = self.pool(features['1'])
        xc2 = self.classifier2(xl2)

        ####################################################################

        ####################################################################


        tmp3 = collections.OrderedDict()
        tmp3['0'] = features['2']
        rpn3_out, _ = self.rpn(images, tmp3)
        roi_in3 = []
        for i in rpn3_out:
            roi_in3.append(i[0:10,:])

        # #########vis
        # show_box = []
        # for i in range(show.size()[0]):
        #     show_feature = show[i]
        #     show_box.append(rpn3_out[i][0:5, :])
        #     self.vis_box(show_feature, show_box)
        #     show_box.clear()
        ###############################
        #roi3_out = self.roialign3(features['1'], list(roi_in3))
        roi3_out = self.single_cat(roi_in3, features['2'], self.roialign3)
        features['2'] = features['2'] + roi3_out
        xl3 = self.pool(features['2'])
        xc3 = self.classifier3(xl3)

        #####################################################################
        x_concat = torch.cat((xl1, xl2, xl3), -1)
        #print(xl3.size())# [C,1024]
        #print(x_concat.size())[C,3072]

        x_concat = self.classifier_concat(x_concat)

        # self.feature1 = xl1
        # self.feature2 = xl2
        # self.feature3 = xl3
        return xc1, xc2, xc3, x_concat
    def vis_box(self,image_tensor, boxes, out_name=None):
      global nums
      image = image_tensor.cpu()
      image_mean = [0.485, 0.456, 0.406]
      image_std = [0.229, 0.224, 0.225]
      mean = torch.as_tensor(image_tensor).cpu()
      std = torch.as_tensor(image_std)
      image = image * std[:,None, None] + mean[:,None, None]
      image = image_tensor.permute((1,2,0))
      image = image * 255.0
      image = image.cpu().numpy()
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      for boxs in boxes:
        boxs = boxs.cpu().numpy()
        boxs = np.int0(boxs)
        #box = [int(i) for i in box]
        for box in boxs:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            cv2.rectangle(image, p1, p2,(0,255,0), 2)
      if out_name is None:
        cv2.imwrite("result/r1/"+str(nums)+"vis_box.png", image)
        # plt.imshow(image.astype('uint8'))
        # plt.axis('off')
        # plt.show()
      else:
        cv2.imwrite(out_name, image)
      nums += 1

    def single_cat(self, rpn_out, features, roialing):
        ### rpn_out list rpn out len = batch  [k,4] k= top boxs  number
        ###
        a = []
        listcat = []
        for index, i in enumerate(rpn_out):
            a.append(i)
            single_feature = torch.unsqueeze(features[index], dim=0)
            out = roialing(single_feature, a)
            a.clear()
            # print(index)
            listcat.append(self.cat_box(out))
            #listcat.append(out)
        cat = self.cat_rpn(listcat)
        return cat

    def cat_box(self,x):
        # k,c,h,w
        #print("catrpn")			
        #cat = torch.cat((x[0], x[1]), dim=0)
       # print(cat.size())
        #cat = torch.cat((cat, x[2]), dim=0)
       # cat = torch.cat((cat, x[3]), dim=0)
       # cat = torch.cat((cat, x[4]), dim=0)
        cat = x[0]
        # cat = x[0]+x[1]+x[2]+x[3]
        for i in range(1, x.size()[0]):
            cat += x[i]
        # cat = x[0]+x[1]+x[2]+x[3]+x[4]+x[5]
        # for i in range(1,x.size()[0]):
        #     cat = torch.cat((,x[i]),dim=-1)
        #     print(x[i].size())
        cat = torch.unsqueeze(cat, dim=0)
        #cat = self.catrpn_conv(cat)
        #print(cat.size())
        return cat

    def cat_rpn(self,select_boxs):
        ###select_boxs  : list , len = batch, cat everyimage box in one tensor
        ###return tensor
        for index, i in enumerate(select_boxs):
            if index == 0:
                cat = i
            else:
                cat = torch.cat((cat, i), dim=0)
        return cat



    def get_params(self):
        new_layers, old_layers = self.features.get_params()
        new_layers += list(self.conv_block1.parameters()) + \
                      list(self.conv_block2.parameters()) + \
                      list(self.conv_block3.parameters()) + \
                      list(self.classifier1.parameters()) + \
                      list(self.classifier2.parameters()) + \
                      list(self.classifier3.parameters()) + \
                      list(self.classifier_concat.parameters())
        return new_layers, old_layers
    def get_feature_params(self, x):
        show = x.clone()
        fm1, fm2, fm3 = self.features(x)

        #########################################
        ##### cross-level attention #############
        #########################################

        att1 = self.conv_block1(fm1)
        att2 = self.conv_block2(fm2)
        att3 = self.conv_block3(fm3)

        new_d1_from2, new_d2_from1 = self.inter(att1, att2)  # 1 2
        new_d1_from3, new_d3_from1 = self.inter(att1, att3)  # 1 3
        new_d2_from3, new_d3_from2 = self.inter(att2, att3)  # 2 3

        gamma = HyperParams['gamma']
        att1 = att1 + gamma * (new_d1_from2 + new_d1_from3)
        att2 = att2 + gamma * (new_d2_from1 + new_d2_from3)
        att3 = att3 + gamma * (new_d3_from1 + new_d3_from2)
        # print(att1.size())(C , 1024, 32, 32)
        # print(att2.size())(C , 1024, 16, 16)
        # print(att3.size())(C , 1024, 8, 8)
        ####################
        d = collections.OrderedDict()
        d['0'] = att1
        d['1'] = att2
        d['2'] = att3
        features = self.FPN(d)

        batchs = att1.size()[0]
        images = []
        for i in range(batchs):
            images.append([300, 300])
        images = ImageList(features['0'], images)

        tmp1 = collections.OrderedDict()
        tmp1['0'] = features['0']
        rpn1_out, _ = self.rpn(images, tmp1)
        roi_in1 = []
        for i in rpn1_out:
            roi_in1.append(i[0:5, :])
        show_box = []
        for i in range(show.size()[0]):
            show_feature = show[i]
            show_box.append(rpn1_out[i][0:5, :])
            self.vis_box(show_feature, show_box)
            show_box.clear()
        # roi1_out = self.single_cat(roi_in1, features['0'],self.roialign1)
        roi1_out = self.roialign1(features['0'], list(roi_in1))
        features['0'] = features['0']+roi1_out

        tmp2 = collections.OrderedDict()
        tmp2['0'] = features['1']

        rpn2_out, _ = self.rpn(images, tmp2)
        roi_in2 = []
        for i in rpn2_out:
            roi_in2.append(i[0:5, :])

        roi2_out = self.roialign2(features['1'], list(roi_in2))
        # print(features['1'].size())
        # roi2_out = self.single_cat(roi_in2, features['1'], self.roialign2)
        # #print(roi2_out.size())
        features['1'] = features['1']+ roi2_out

        ####################################################################

        ####################################################################

        tmp3 = collections.OrderedDict()
        tmp3['0'] = features['2']
        rpn3_out, _ = self.rpn(images, tmp3)
        roi_in3 = []
        for i in rpn3_out:
            roi_in3.append(i[0:5, :])

        #########vis
        show_box = []
        for i in range(show.size()[0]):
            show_feature = show[i]
            show_box.append(rpn3_out[i][0:5, :])
            self.vis_box(show_feature, show_box)
            show_box.clear()
        ###############################
        roi3_out = self.roialign3(features['1'], list(roi_in3))
        # roi3_out = self.single_cat(roi_in3, features['2'], self.roialign3)
        features['2'] = features['2'] + roi3_out
        return features['0'], features['1'], features['2']
