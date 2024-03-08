import os
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import FBSD
from config import HyperParams

folder_path='/home/hello/twelve/ablation/redata/test'
folder_file_names=os.listdir(folder_path)
for folder_file_name in folder_file_names:
    folder_file_path = os.path.join(folder_path, folder_file_name)
    file_names = os.listdir(folder_file_path)
    for file_name in file_names:
        image_path=os.path.join(folder_file_path,file_name)
        output_folder = "FV/" + image_path.split('/')[7]

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 加载模型，并设置为评估模式
        # model = FBSD(class_num=196, arch='resnet50')
        model = FBSD(class_num=200, arch=HyperParams['arch'])
        model = model.cuda()
        model.load_state_dict(torch.load('addfpn_rpn_resnet50_output_98.1203/best_model.pth'))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # 读取图像并预处理
        image = Image.open(image_path)
        image_processed = transform(image).unsqueeze(0)
        image_processed = image_processed.cuda()

        # 使用网络模型提取特征
        with torch.no_grad():
            output_1, output_2, output_3 = model.get_feature_params(image_processed)

        # 将特征图转换为彩色图像
        features_np1 = output_1.squeeze(0).cpu().numpy()
        features_sum1 = np.sum(features_np1, axis=0)
        features_np2 = output_2.squeeze(0).cpu().numpy()
        features_sum2 = np.sum(features_np2, axis=0)
        features_np3 = output_3.squeeze(0).cpu().numpy()
        features_sum3 = np.sum(features_np3, axis=0)

        # 对特征值进行缩放以提高可视化效果
        alpha = 1
        scaled_features_sum1 = alpha * (features_sum1 - np.min(features_sum1))
        features_normalized1 = scaled_features_sum1 / np.max(scaled_features_sum1)
        scaled_features_sum2 = alpha * (features_sum2 - np.min(features_sum2))
        features_normalized2 = scaled_features_sum2 / np.max(scaled_features_sum2)
        scaled_features_sum3 = alpha * (features_sum3 - np.min(features_sum3))
        features_normalized3 = scaled_features_sum3 / np.max(scaled_features_sum3)

        # 将特征图缩放到原始图像大小并转换为BGR格式
        image_resized = np.array(image.resize((224, 224)))
        features_resized1 = cv2.resize(features_normalized1, (224, 224))
        feature_image1 = cv2.applyColorMap((features_resized1 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        feature_image_bgr1 = cv2.cvtColor(feature_image1, cv2.COLOR_RGB2BGR)
        features_resized2 = cv2.resize(features_normalized2, (224, 224))
        feature_image2 = cv2.applyColorMap((features_resized2 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        feature_image_bgr2 = cv2.cvtColor(feature_image2, cv2.COLOR_RGB2BGR)
        features_resized3 = cv2.resize(features_normalized3, (224, 224))
        feature_image3 = cv2.applyColorMap((features_resized3 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        feature_image_bgr3 = cv2.cvtColor(feature_image3, cv2.COLOR_RGB2BGR)

        # 将特征图与原始图像叠加
        overlay_alpha = 0.5
        combined_image1 = cv2.addWeighted(image_resized, 1 - overlay_alpha, feature_image_bgr1, overlay_alpha, 0)
        combined_image2 = cv2.addWeighted(image_resized, 1 - overlay_alpha, feature_image_bgr2, overlay_alpha, 0)
        combined_image3 = cv2.addWeighted(image_resized, 1 - overlay_alpha, feature_image_bgr3, overlay_alpha, 0)

        image_name = image_path.split("/")[8]
        image_name = os.path.splitext(image_name)[0]
        combined_image_path1 = os.path.join(output_folder, image_name + '_1.jpg')
        combined_image_path2 = os.path.join(output_folder, image_name + '_2.jpg')
        combined_image_path3 = os.path.join(output_folder, image_name + '_3.jpg')
        cv2.imwrite(combined_image_path1, cv2.cvtColor(combined_image1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(combined_image_path2, cv2.cvtColor(combined_image2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(combined_image_path3, cv2.cvtColor(combined_image3, cv2.COLOR_RGB2BGR))

        print(f"叠加后的图像已成功保存到：{output_folder}")
# 设置数据集图片路径和输出文件夹路径
# image_path = "/home/hello/twelve/ablation/redata/test/CC/BSI-CFM56-20201323-2.JPG"
# output_folder = "FeatureVisualization/"+image_path.split('/')[7]
#
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
#
# # 加载模型，并设置为评估模式
# # model = FBSD(class_num=196, arch='resnet50')
# model = FBSD(class_num=200, arch=HyperParams['arch'])
# model = model.cuda()
# model.load_state_dict(torch.load('addfpn_rpn_resnet50_output_98.1203/best_model.pth'))
# model.eval()
#
# # 定义图像预处理
# # transform = transforms.Compose([
# #     transforms.Resize(256),
# #     transforms.CenterCrop(224),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# # ])
# # transform = transforms.Compose([
# #         transforms.Resize((550, 550)),
# #         transforms.CenterCrop(448),
# #         transforms.ToTensor(),
# #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# #     ])
# transform = transforms.Compose([
#     transforms.Resize((300, 300)),
#     transforms.CenterCrop(256),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
#
# # 读取图像并预处理
# image = Image.open(image_path)
# image_processed = transform(image).unsqueeze(0)
# image_processed=image_processed.cuda()
#
# # 使用网络模型提取特征
# with torch.no_grad():
#     output_1, output_2, output_3 = model.get_feature_params(image_processed)
#
# # 将特征图转换为彩色图像
# features_np1 = output_1.squeeze(0).cpu().numpy()
# features_sum1 = np.sum(features_np1, axis=0)
# features_np2 = output_2.squeeze(0).cpu().numpy()
# features_sum2 = np.sum(features_np2, axis=0)
# features_np3 = output_3.squeeze(0).cpu().numpy()
# features_sum3 = np.sum(features_np3, axis=0)
#
# # 对特征值进行缩放以提高可视化效果
# alpha = 1
# scaled_features_sum1 = alpha * (features_sum1 - np.min(features_sum1))
# features_normalized1 = scaled_features_sum1 / np.max(scaled_features_sum1)
# scaled_features_sum2 = alpha * (features_sum2 - np.min(features_sum2))
# features_normalized2 = scaled_features_sum2 / np.max(scaled_features_sum2)
# scaled_features_sum3 = alpha * (features_sum3 - np.min(features_sum3))
# features_normalized3 = scaled_features_sum3 / np.max(scaled_features_sum3)
#
# # 将特征图缩放到原始图像大小并转换为BGR格式
# image_resized = np.array(image.resize((224, 224)))
# features_resized1 = cv2.resize(features_normalized1, (224, 224))
# feature_image1 = cv2.applyColorMap((features_resized1 * 255).astype(np.uint8), cv2.COLORMAP_JET)
# feature_image_bgr1 = cv2.cvtColor(feature_image1, cv2.COLOR_RGB2BGR)
# features_resized2 = cv2.resize(features_normalized2, (224, 224))
# feature_image2 = cv2.applyColorMap((features_resized2 * 255).astype(np.uint8), cv2.COLORMAP_JET)
# feature_image_bgr2 = cv2.cvtColor(feature_image2, cv2.COLOR_RGB2BGR)
# features_resized3 = cv2.resize(features_normalized3, (224, 224))
# feature_image3 = cv2.applyColorMap((features_resized3 * 255).astype(np.uint8), cv2.COLORMAP_JET)
# feature_image_bgr3 = cv2.cvtColor(feature_image3, cv2.COLOR_RGB2BGR)
#
# # 将特征图与原始图像叠加
# overlay_alpha = 0.5
# combined_image1 = cv2.addWeighted(image_resized, 1 - overlay_alpha, feature_image_bgr1, overlay_alpha, 0)
# combined_image2 = cv2.addWeighted(image_resized, 1 - overlay_alpha, feature_image_bgr2, overlay_alpha, 0)
# combined_image3 = cv2.addWeighted(image_resized, 1 - overlay_alpha, feature_image_bgr3, overlay_alpha, 0)
#
# # 保存叠加后的图像到输出文件夹
# # combined_image_path1 = os.path.join(output_folder, image_path.split("/")[8] + '_1')
# # combined_image_path2 = os.path.join(output_folder, image_path.split("/")[8] + '_2')
# # combined_image_path3 = os.path.join(output_folder, image_path.split("/")[8] + '_3')
# image_name= image_path.split("/")[8]
# image_name=os.path.splitext(image_name)[0]
# combined_image_path1 = os.path.join(output_folder, image_name + '_1.jpg')
# combined_image_path2 = os.path.join(output_folder, image_name + '_2.jpg')
# combined_image_path3 = os.path.join(output_folder, image_name + '_3.jpg')
# cv2.imwrite(combined_image_path1, cv2.cvtColor(combined_image1, cv2.COLOR_RGB2BGR))
# cv2.imwrite(combined_image_path2, cv2.cvtColor(combined_image2, cv2.COLOR_RGB2BGR))
# cv2.imwrite(combined_image_path3, cv2.cvtColor(combined_image3, cv2.COLOR_RGB2BGR))
#
# print(f"叠加后的图像已成功保存到：{output_folder}")