import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.nn import CrossEntropyLoss
import math
from transformers import RobertaModel,BertModel,AlbertModel,ElectraModel,ViTModel,SwinModel,DeiTModel,ConvNextModel
from Module.SemanticAlignment import visual_seq_feature, Semantic_AlignmentModel
import sys
import spacy
import numpy as np
import logging
from torch_geometric.nn import GCNConv


class HybridSequenceDenoisingBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, conv_kernel_size=3):
        super().__init__()
        # 1. 1D卷积处理序列局部特征（替代2D卷积）
        self.conv_block = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2,
                      groups=embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),  # 1x1卷积调整通道
            nn.BatchNorm1d(embed_dim)
        )

        # 2. 自注意力层（保持全局依赖）
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 3. 残差连接与归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 输入形状: [B, Seq_Len, Embed_Dim] = [6, 197, 768]
        B, N, C = x.shape

        # --- Step 1: 卷积处理（需调整维度）---
        x_conv = self.conv_block(x.permute(0, 2, 1))  # [B, C, Seq_Len]
        x_conv = x_conv.permute(0, 2, 1)  # [B, Seq_Len, C]

        # --- Step 2: 自注意力处理 ---
        x_attn, _ = self.self_attn(x_conv, x_conv, x_conv)  # [B, Seq_Len, C]
        x_attn = self.norm1(x_attn + x_conv)  # 残差连接

        # --- Step 3: 最终残差 ---
        x_out = self.norm2(x_attn + x)
        return x_out


class ViTDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.hybrid_block = HybridSequenceDenoisingBlock()
        # 分类标记投影层（可选）
        self.cls_proj = nn.Linear(768, 768)

    def forward(self, vit_features):
        # 输入形状: [6, 197, 768]
        # 分离分类标记和图像块特征
        cls_token = vit_features[:, 0:1, :]  # [6, 1, 768]
        patch_tokens = vit_features[:, 1:, :]  # [6, 196, 768]

        # 对图像块特征去噪
        denoised_patches = self.hybrid_block(patch_tokens)  # [6, 196, 768]

        # 重新合并分类标记（可添加投影）
        denoised_cls = self.cls_proj(cls_token)  # [6, 1, 768]
        output = torch.cat([denoised_cls, denoised_patches], dim=1)  # [6, 197, 768]
        return output

class DynamicGraphConstructor(nn.Module):
    def __init__(self, hidden_dim=768, k_neighbors=5):
        super().__init__()
        self.k = k_neighbors
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        edge_indices = []
        for b in range(batch_size):
            # 计算节点相似度
            sim_matrix = torch.matmul(x[b], x[b].T)  # [seq_len, seq_len]
            # 选择top-k邻居
            _, topk_indices = torch.topk(sim_matrix, self.k, dim=1)
            # 构建边索引
            src = torch.repeat_interleave(torch.arange(seq_len, device=x.device), self.k)
            dst = topk_indices.view(-1)
            # 双向边
            edge_index = torch.stack([torch.cat([src, dst]),
                                      torch.cat([dst, src])], dim=0)
            # 去重
            edge_index = torch.unique(edge_index, dim=1)
            edge_indices.append(edge_index)
        return edge_indices


class GCNTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        # GCN组件
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim)

        # Transformer组件
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 门控融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        # 调整维度顺序以适应GCN
        batch_size = x.size(1)
        x_gcn = x.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]

        # GCN处理
        gcn_out = []
        for b in range(batch_size):
            gcn_out.append(self.gcn_conv(x_gcn[b], edge_index[b]))
        gcn_out = torch.stack(gcn_out, dim=1)  # [seq_len, batch_size, hidden_dim]

        # Transformer处理
        attn_out, _ = self.self_attn(x, x, x)  # [seq_len, batch_size, hidden_dim]

        # 门控融合
        combined = torch.cat([gcn_out, attn_out], dim=-1)  # [seq_len, batch_size, 2*hidden_dim]
        gate = self.fusion_gate(combined)  # [seq_len, batch_size, hidden_dim]
        fused = gate * gcn_out + (1 - gate) * attn_out
        # 残差连接
        output = self.norm2(fused + x)
        return output


class GCNTransformer(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()

        self.graph_builder = DynamicGraphConstructor()
        self.layers = nn.ModuleList([
            GCNTransformerLayer() for _ in range(num_layers)
        ])

        # 维度适配器（保持输出维度与输入一致）
        self.output_proj = nn.Linear(768, 768)

    def forward(self, x):

        # 调整维度顺序以适应Transformer
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]

        # 动态构建图结构
        edge_indices = self.graph_builder(x.permute(1, 0, 2))  # 恢复batch_first

        # 逐层处理
        for layer in self.layers:
            x = layer(x, edge_indices)

        # 恢复原始维度顺序
        output = x.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]

        # 维度调整保证输出一致
        output = self.output_proj(output)

        return output


logger = logging.getLogger(__name__)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias




class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, str)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Attention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        # 线性映射层定义
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        # 最终映射层
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, score_matrix):
        # 执行线性映射以获取 Q, K, V
        Q = self.query_linear(query)  # (4, 60, 768)
        K = self.key_linear(key)      # (4, 197, 768)
        V = self.value_linear(value)  # (4, 197, 768)

        # 根据得分矩阵调整注意力分数
        # Q: (4, 60, 768), K: (4, 197, 768)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) + score_matrix  # (4, 60, 197)

        # 计算 softmax 归一化的注意力权重
        attention_weights = F.softmax(attention_scores / torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32)), dim=-1)  # (4, 60, 197)

        # 加权平均得到输出
        attention_output = torch.matmul(attention_weights, V)  # (4, 60, 768)

        # 最终线性映射
        attention_output = self.final_linear(attention_output)  # (4, 60, 768)

        return attention_output


class MVATModel(nn.Module):
    def __init__(self,config1,config2,text_num_labels,alpha,beta,text_model_name="roberta",image_model_name='vit', dropout_rate=0.1, num_heads=8, d_model=768):
        super().__init__()
        if text_model_name == 'roberta':
            self.roberta = RobertaModel(config1,add_pooling_layer=False)
        elif text_model_name == 'bert':
            self.bert = BertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'albert':
            self.albert = AlbertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'electra':
            self.electra = ElectraModel(config1)
        if image_model_name == 'vit':
            self.vit = ViTModel(config2)
        elif image_model_name == 'swin':
            self.swin = SwinModel(config2)
        elif image_model_name == 'deit':
            self.deit = DeiTModel(config2)
        elif image_model_name == 'convnext':
            self.convnext = ConvNextModel(config2)
        self.alpha = alpha
        self.beta = beta
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.config1 = config1
        self.config2 = config2
        self.text_num_labels = text_num_labels
        self.image_text_cross = MultiHeadAttention(8, config1.hidden_size,config1.hidden_size,config1.hidden_size)
        self.dropout = nn.Dropout(config1.hidden_dropout_prob)
        self.loss_fct = CrossEntropyLoss()
        self.classifier1 = nn.Linear(config1.hidden_size, self.text_num_labels)
        self.classifier0= nn.Linear(config1.hidden_size, self.text_num_labels)
        self.CRF = CRF(self.text_num_labels, batch_first=True)
        self.attention = Attention(num_heads=8, d_model=768)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.intermediate = BertIntermediate(config1)
        self.output = BertOutput(config1)
        self.LayerNorm_top = BertLayerNorm(config1.hidden_size, eps=config1.layer_norm_eps)
        self.GCNTransformer = GCNTransformer(num_layers=3)
        self.ViTDenoiser = ViTDenoiser()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                pixel_values=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                image_labels=None,
                head_mask=None,
                cross_labels=None,
                return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config1.use_return_dict
        if self.text_model_name == 'bert':
            text_outputs = self.bert(input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        elif self.text_model_name == 'roberta':
            text_outputs = self.roberta(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'albert':
            text_outputs = self.albert(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'electra':
            text_outputs = self.electra(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        else:
            text_outputs=None
        if self.image_model_name == 'vit':
            image_outputs = self.vit(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'swin':
            image_outputs = self.swin(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'deit':
            image_outputs = self.deit(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'convnext':
            image_outputs = self.convnext(pixel_values)
        else:
            image_outputs=None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        semantic_model = Semantic_AlignmentModel().to(device)

        text_last_hidden_states = text_outputs["last_hidden_state"]
        image_last_hidden_states = image_outputs["last_hidden_state"]
        # image_last_hidden_states = self.ViTDenoiser(image_last_hidden_states)
        text_last_hidden_states = self.GCNTransformer(text_last_hidden_states)
        text_last_hidden_states = text_last_hidden_states.to(device)
        image_last_hidden_states = image_last_hidden_states.to(device)
        #print('image_last_hidden_states.shape:', image_last_hidden_states.shape)
        score_matrix2 = semantic_model(text_last_hidden_states, image_last_hidden_states)
        # print(score_matrix2.shape)
        #print(f"text_output shape: {text_last_hidden_states.shape}")
        # print(f"visual_output shape: {image_last_hidden_states.shape}")
        # print(score_matrix2.shape)  #torch.Size([4, 60, 197])


        image_text_cross_attention = self.attention(text_last_hidden_states, image_last_hidden_states,
                                                              image_last_hidden_states, score_matrix2)
        attention_output = self.dropout(image_text_cross_attention)
        out1 = self.layernorm1(text_last_hidden_states + attention_output)

        # cross_crf_loss
        image_text_cross_attention, _ = self.image_text_cross(text_last_hidden_states, image_last_hidden_states,
                                                              image_last_hidden_states)
        #print(f"image_text_cross_attention shape: {image_text_cross_attention.shape}")
        # image_text_cross_attentionshape: torch.Size([4, 60, 768])
        # 文本置信度**********************
        sequence_output1 = self.dropout(text_last_hidden_states)
        text_token_logits = self.classifier1(sequence_output1)
        aux_output = nn.Softmax(dim=-1)(text_token_logits)
        c = torch.sum(aux_output * aux_output, dim=-1)
        gate_value = c.unsqueeze(-1)
        reverse_gate_value = torch.neg(gate_value).add(1)
        gated_converted_att_vis_embed = torch.mul(reverse_gate_value, image_text_cross_attention)
        gated_main_addon_sequence_output = torch.mul(gate_value, out1)
        weighted_add_output = torch.add(gated_main_addon_sequence_output, gated_converted_att_vis_embed)
        weighted_add_norm_output = self.LayerNorm_top(weighted_add_output+text_last_hidden_states)
        intermediate_output = self.intermediate(weighted_add_norm_output)
        layer_output = self.output(intermediate_output, weighted_add_norm_output)
        final_output = self.LayerNorm_top(layer_output + out1)
        # 文本置信度**********************
        out1 = self.LayerNorm_top(image_text_cross_attention )
        cross_logits = self.classifier0(out1)
        mask = (labels != -100)
        mask[:,0] = 1
        # print(cross_logits.shape, cross_labels.shape)
        cross_crf_loss = -self.CRF(cross_logits, cross_labels, mask=mask) / 10
        # word patch align
        batch_size, image_len, _ = image_last_hidden_states.shape
        _, text_len, _ = text_last_hidden_states.shape
        text_pad = (attention_mask == 1).clone().detach()
        image_pad = torch.zeros(batch_size, image_len, dtype=torch.bool, device=attention_mask.device)
        ot_dist = optimal_transport_dist(text_last_hidden_states, image_last_hidden_states,text_pad,image_pad)
        # print(text_pad.shape)
        # print(image_pad.shape)

        word_region_align_loss = ot_dist.mean()
        # text_loss
        sequence_output1 = self.dropout(text_last_hidden_states)
        text_token_logits = self.classifier1(sequence_output1)

        # word_region_align_loss = ot_dist.masked_select(targets == 0)
        # getTextLoss: CrossEntropy
        text_loss = self.loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1))
        loss = cross_crf_loss + self.beta * word_region_align_loss + self.alpha * text_loss
        # loss = text_loss

        # end train
        return {"loss":loss,
            "logits":text_token_logits,
            "cross_logits": cross_logits,
                }


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False, attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            # self.dropout2 = nn.Dropout(dropout2)
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None, dec_self=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        if hasattr(self, 'dropout2'):
            q = self.dropout2(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        if hasattr(self, 'fc'):
            output = self.fc(output)

        if hasattr(self, 'dropout'):
            output = self.dropout(output)

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
            # self.softmax = BottleSoftmax()
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            # attn = attn.masked_fill(attn_mask, -np.inf)
            attn = attn.masked_fill(attn_mask, -1e6)

        if stop_sig:
            print('**')
            stop()

        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss



def cost_matrix_cosine(x, y, eps=1e-5):
    """ Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device
                     ).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(
        b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device
                       ) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2)/beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(txt_emb, img_emb, txt_pad, img_pad,
                           beta=0.5, iteration=50, k=1):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    # print('txt_emb.shape:', txt_emb.shape)
    # print('img_emb.shape', img_emb.shape)
    # print('txt_pad', txt_pad.shape)
    # print('img_pad', img_pad.shape)
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)

    T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad,
             beta, iteration, k)
    distance = trace(cost.matmul(T.detach()))
    # print('distance.shape:', distance.shape)
    return distance
