import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn
from PIL import Image, ImageDraw, ImageFont
from BaseModel.SceneGraph import Image_traget_detection, build_scene_graph, scene_draw, extract_features
from BaseModel.ViTProcess import preprocess_image
from transformers import ViTModel, ViTConfig
from PIL import Image
import numpy as np
import requests
from torchvision import transforms

#using vit model process image
vit_model = ViTModel.from_pretrained('D:/pytorch project/DTCA-main/vit-base-patch16-224-in21k')

def visual_seq_feature(image_path):
    input_tensor = preprocess_image(image_path)
    # # 获取图片的表征
    with torch.no_grad():
        outputs = vit_model(pixel_values=input_tensor)
    last_hidden_states = outputs.last_hidden_state
    output_tensor = last_hidden_states.squeeze(0)
    return output_tensor


#Semantic Alignment compute
class Semantic_AlignmentModel(nn.Module):
    def __init__(self):
        super(Semantic_AlignmentModel, self).__init__()
        self.W1 = nn.Parameter(torch.randn(768, 768))  # 定义 W1 为可训练的参数
        self.W2 = nn.Parameter(torch.randn(768, 768))  # 定义 W2 为可训练的参数
        self.W3 = nn.Parameter(torch.randn(768, 768))  # 定义 W3 为可训练的参数

    def forward(self, text_output, visual_output):
        # 确保 text_output 和 visual_output 在同一设备上
        text_output = text_output.to(self.W1.device)
        visual_output = visual_output.to(self.W1.device)
        batch_size = text_output.size(0)
        # text_seq_len = text_output.size(1)
        # visual_seq_len = visual_output.size(1)
        # 计算对齐矩阵 C
        # 首先对 text_output 和 visual_output 进行线性变换
        text_transformed = torch.matmul(text_output, self.W1)  # (4, 60, 768)
        visual_transformed = torch.matmul(visual_output, self.W1.T)  # (4, 197, 768)
        # 计算每对文本和视觉特征的点积
        C = torch.matmul(text_transformed, visual_transformed.permute(0, 2, 1))  # (4, 60, 197)
        C = torch.tanh(C)
        # 计算相关矩阵 M
        W2_text = torch.matmul(text_output, self.W2).unsqueeze(2)  # (4, 60, 1, 768)
        W3_visual = torch.matmul(visual_output, self.W3.T).unsqueeze(1)  # (4, 1, 197, 768)
        C_expanded = C.unsqueeze(-1)  # (4, 60, 197, 1)
        M = torch.tanh(W2_text + W3_visual * C_expanded)  # (4, 60, 197, 768)
        # 计算 text score matrix
        score_matrix = []
        for i in range(batch_size):
            Mi = M[i, :, :, :]  # 提取三维矩阵 M 的第 i 个样本，形状为 (60, 197, 768)
            ti = text_output[i, :, :].unsqueeze(1)  # 提取文本的第 i 个样本，形状为 (60, 1, 768)
            dot_product = torch.matmul(ti, Mi.permute(0, 2, 1))  # 矩阵乘法，得到形状为 (60, 197)
            score_matrix.append(dot_product.squeeze(1))  # 将结果添加到 score_matrix 中
        # 将结果矩阵组合起来，得到形状为 (4, 60, 197) 的矩阵
        score_matrix = torch.stack(score_matrix)
        return score_matrix




# # #model fit train
# image_path = 'E:/PythonProject2/VLHAFigure/16_05_13_757.jpg'
# sentence = 'Franklin baseball team hosting USA Baseball 12U camps'
#
# image_feature = visual_seq_feature(image_path)
# # print(image_feature.shape)   #torch.Size([197, 768])
# text_feature = BERT_Embedding(sentence)
# # print(text_feature.shape)    #torch.Size([10, 768])
# # 实例化模型
# semantic_model = Semantic_AlignmentModel()
#
# # 使用模型进行前向传播计算
# score_matrix = semantic_model(text_feature, image_feature)
# print(score_matrix.shape)  #(8,197)
#
# import matplotlib.pyplot as plt
# # 归一化到 [0, 1] 范围
# min_val = score_matrix.min().item()
# max_val = score_matrix.max().item()
# normalized_tensor = (score_matrix - min_val) / (max_val - min_val)
# # 将张量转换为numpy数组
# tensor_np = normalized_tensor.detach().numpy()
#
# # 定义y轴刻度标签
# y_labels = ["Franklin", "baseball", "team", "hosting", "USA", "Baseball", "12U", "camps"]
#
#
# # 绘制热力图
# plt.figure(figsize=(10, 6))  # 设置图形大小
# # 使用imshow绘制热力图，选择颜色映射（cmap）为默认的热图（'hot'）
# plt.imshow(tensor_np, cmap='jet', aspect='auto')
# # 添加颜色条
# plt.colorbar()
# # 添加标题和标签
# # plt.title('Heatmap of Tensor')
# # plt.xlabel('Image patches')
# # plt.ylabel('Text words')
# # 设置y轴刻度标签
# plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)
#
# # 显示图形
# plt.show()