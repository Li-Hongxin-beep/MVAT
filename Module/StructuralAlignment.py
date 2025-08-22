import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn
from PIL import Image

from BaseModel.GCNmodel import GCNModel
from BaseModel.SceneGraph import (
    Image_traget_detection,
    build_scene_graph,
    extract_features,
    build_graph_structure
)
from torch.nn.utils.rnn import pad_sequence
from BaseModel.SyntacticGraph import BiAffine, BERT_Embedding

# 视觉 GCN
visual_model = GCNModel(input_size=2048, hidden_size=768, num_layers=2)
# 文本 GCN
text_model   = GCNModel(input_size=768,  hidden_size=768, num_layers=2)

# ------------------------------------------------------------------
# 1. 视觉图计算（单张图）
# ------------------------------------------------------------------
def Visual_graph_compute(image_path, threshold, visual_model, device=None):
    """
    image_path: str
    threshold : float
    device    : torch.device，缺省自动跟随 visual_model
    """
    if device is None:
        device = next(visual_model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    outputs = Image_traget_detection(image, threshold)
    scene_graph = build_scene_graph(outputs, threshold)

    node_feature = extract_features(image, scene_graph)        # [N, 2048]
    nodes, edge_tensor = build_graph_structure(scene_graph, node_feature)

    node_feature = node_feature.to(device)
    edge_tensor  = edge_tensor.to(device)

    with torch.no_grad():
        output = visual_model(node_feature, edge_tensor)       # [N, 768]
    return output


def single_Syntax_Graph_compute(sentence, text_model, device=None):
    if device is None:
        device = next(text_model.parameters()).device

    # 1️⃣ 保证是 str，而不是 list[str]
    if isinstance(sentence, list):
        sentence = ' '.join(sentence)

    text_graph = BiAffine(sentence).to(device)
    word_feature = BERT_Embedding(sentence).to(device)
    output = text_model(word_feature, text_graph)
    return output

# 供 DTCAModel 调用的批处理封装
def Syntax_Graph_compute(sentences, text_model, device=None):
    """
    sentences: List[str]  batch
    返回      padded_feat, mask
    """
    if device is None:
        device = next(text_model.parameters()).device
    graphs = [single_Syntax_Graph_compute(sent, text_model, device) for sent in sentences]
    lengths = [g.size(0) for g in graphs]
    padded = torch.nn.utils.rnn.pad_sequence(graphs, batch_first=True)
    mask   = torch.arange(padded.size(1)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    return padded, mask.to(device)

#structural alignment compute
class AlignmentModel(nn.Module):
    def __init__(self):
        super(AlignmentModel, self).__init__()
        self.W1 = nn.Parameter(torch.randn(768, 768))  # 定义 W1 为可训练的参数
        self.W2 = nn.Parameter(torch.randn(768, 768))  # 定义 W2 为可训练的参数
        self.W3 = nn.Parameter(torch.randn(768, 768))  # 定义 W3 为可训练的参数
        self.linear = nn.Linear(512, 768)  # 定义用于 L 函数的线性层

    def forward(self, text_output, visual_output):
        # 计算对齐矩阵 C
        C = torch.tanh(torch.matmul(torch.matmul(text_output, self.W1), visual_output.T))  # (10, 4)

        # 计算相关矩阵 M, 维度扩张2dim--->3dim
        W2_text = torch.matmul(text_output, self.W2).unsqueeze(1)  # (10, 1, 256)
        W3_visual = torch.matmul(visual_output, self.W3.T).unsqueeze(0)  # (1, 4, 256)
        C_expanded = C.unsqueeze(-1)  # (10, 4, 1)
        M = torch.tanh(W2_text + W3_visual * C_expanded)  # (10, 4, 256)

        # 计算 text score matrix
        score_matrix = []
        num_rows = M.size(0)
        for i in range(num_rows):
            Mi = M[i, :, :]   # 提取三维矩阵 M 的第 i 行，形状为 (4, 256)

            ti = text_output[i, :].unsqueeze(0)   # 提取文本的第 i 行，形状为 (1,256)

            # 进行点积计算
            dot_product = torch.matmul(ti, Mi.T)  # 将 Mi 转置后进行矩阵乘法，得到形状为 (1, 4)
            score_matrix.append(dot_product.squeeze(0))  # 将结果添加到 result_matrix 中，并调整形状为 (4,)

        # 将结果矩阵组合起来，得到形状为 (10, 4) 的矩阵
        score_matrix = torch.stack(score_matrix)
        return score_matrix


# # model fit train
# image_path = 'E:/PythonProject2/VLHA/Datasets/16_05_23_915.jpg'
# threshold = 0.6  # 阈值
# sentence = 'David Gilmour and Roger Waters playing table football'

# visual_output = Visual_graph_compute(image_path, threshold, visual_model)
# print(visual_output.shape)   #(4, 768)
# text_output = Syntax_Graph_compute(sentence, text_model)
# print(text_output.shape)  #(10, 768)
#
#
# # 实例化模型
# model = AlignmentModel()
# Score_matrix = model(text_output, visual_output)
# print(Score_matrix)  #(10,4)
