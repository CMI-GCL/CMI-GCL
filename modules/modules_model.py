import torch
from torch import nn
import timm
from transformers import BertModel,BertConfig,BertTokenizer

import config as CFG
import  torch.nn.functional as F
from modules.modules_node import GraphConvolution,GraphAttentionLayer,GraphSageConv

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.image_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )   # 加载你需要的预训练模型
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            # self.model = DistilBertModel.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
        else:
            # self.model = DistilBertModel(config=DistilBertConfig())
            self.model = BertModel(config=BertConfig())
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectNode(nn.Module):
    def __init__(self,embedding_dim,projectnode_dim):
        super().__init__()
        self.projectnode = nn.Linear(embedding_dim,projectnode_dim)

    def forward(self,x):
        projected_node = self.projectnode(x)
        return projected_node



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x



class NodeEncoder(nn.Module):
    def __init__(self):
        super(NodeEncoder, self).__init__()

        # self.gc1 = GraphConvolution(CFG.node_feature_project_dim, CFG.hidden_dim)
        if CFG.data == 'github':
            node_feature_dim = CFG.github_node_feature_dim
        elif CFG.data == 'insta':
            node_feature_dim = CFG.insta_node_feature_dim
        elif CFG.data == 'yelp':
            node_feature_dim = CFG.yelp_node_feature_dim

        self.dropout = CFG.node_dropout
        if CFG.node_backbone == 'gcn':
            self.gc1 = GraphConvolution(node_feature_dim, CFG.hidden_dim)
            self.gc2 = GraphConvolution(CFG.hidden_dim, CFG.out_dim)
            # self.gc3 = GraphConvolution(CFG.out_dim, CFG.nclass)

        elif CFG.node_backbone == 'gat':

            self.attentions = [GraphAttentionLayer(node_feature_dim, CFG.hidden_dim, dropout=self.dropout, alpha=CFG.alpha, concat=True) for _ in
                               range(CFG.nheads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)

            self.out_att = GraphAttentionLayer(CFG.hidden_dim * CFG.nheads, CFG.out_dim, dropout=CFG.dropout, alpha=CFG.alpha, concat=False)
        elif CFG.node_backbone == 'sage':
            self.sage1 = GraphSageConv(node_feature_dim, CFG.hidden_dim)
            self.sage2 = GraphSageConv(CFG.hidden_dim, CFG.out_dim)


    def forward(self, x, adj):
        if CFG.node_backbone == 'gcn':
            x1 = F.relu(self.gc1(x, adj))
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = self.gc2(x1, adj)
            # x3 = self.gc3(x2,adj)

        elif CFG.node_backbone == 'gat':
            x = F.dropout(x, self.dropout, training=self.training)
            x1 = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = F.elu(self.out_att(x1, adj))

        elif CFG.node_backbone == 'sage':

            x1 = F.relu(self.sage1(x, adj))
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = F.relu(self.sage2(x1, adj))
            x2 = F.dropout(x2, self.dropout, training=self.training)

        # return F.log_softmax(x, dim=1)
        return x2


class NodeClassifier(nn.Module):
    def __init__(self):
        super(NodeClassifier, self).__init__()

        self.nodencoder = NodeEncoder()
        self.mlp = nn.Linear(CFG.out_dim, CFG.nclass)
        self.dropout = CFG.dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.nodencoder(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x[:,:])

        return x

