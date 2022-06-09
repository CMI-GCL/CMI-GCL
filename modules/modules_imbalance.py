#!/usr/bin/env python
# encoding: utf-8

import random

from copy import deepcopy
import config as CFG

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# adj, features, labels, idx_train, idx_val, idx_test = load_data()

def Upsample(adj, features, labels, idx_train, portion=1.0):

    num_neg = len(idx_train[(labels == 0.0)[idx_train]])
    num_pos = len(idx_train[(labels == 1.0)[idx_train]])

    num_add = num_neg - num_pos
    idx_pos = idx_train[(labels == 1.0)[idx_train]]
    num_upsample = int(num_add / num_pos)
    adj_row = adj.coalesce().indices()[0, :]
    adj_column = adj.coalesce().indices()[1, :]
    adj_values = adj.coalesce().values()
    adj_new_row,adj_new_col,adj_new_value,idx_add = [],[],[],[]
    num_node = adj.shape[0]
    idx_all = []
    num_upsample = int(num_upsample * portion)
    if CFG.data == 'insta':
        num_upsample = 1
    if num_upsample >= 1:
        for k in range(num_upsample):
            idx_add += idx_pos.tolist()
            if k == 0:
                for j in idx_pos.tolist():

                    idx_ = (adj_row == j).nonzero(as_tuple=False).reshape(1,-1).tolist()[0]
                    idx_all.append(idx_)
                    if len(idx_) == 0:
                        adj_new_row.append(j)
                        adj_new_col.append(j)
                        adj_new_value.append(1)
                        continue

                    adj_new_row += [num_node for i in range(len(idx_))]
                    adj_new_col+=[adj_column[i].tolist()  for i in idx_]
                    adj_new_value += [adj_values[i].tolist() for i in idx_]
                    adj_new_row.append(num_node)
                    adj_new_col.append(num_node)
                    adj_new_value.append(0.5)
                    num_node += 1
            else:
                for i in idx_all:
                    for j in i:
                        adj_new_row.append(num_node)
                        adj_new_col.append(adj_column[j].tolist())
                        adj_new_value.append(adj_values[j].tolist())
                    adj_new_row.append(num_node)
                    adj_new_col.append(num_node)
                    adj_new_value.append(0.01)
                    num_node += 1
                # adj_new_row+=[num_node for i in idx_all for j in i]
                # adj_new_col+=[adj_column[j].tolist() for i in idx_all for j in i]
                # adj_new_value+=[adj_values[j].tolist()  for i in idx_all for j in i]

    else:
        idx_add = torch.tensor(random.sample(idx_pos.tolist(),num_add))

    adj_final_row = adj_row.tolist() +  adj_new_row
    adj_final_column = adj_column.tolist() +  adj_new_col
    adj_final_value = adj_values.tolist() + adj_new_value

    adj_new = torch.sparse_coo_tensor(np.array((adj_final_row, adj_final_column)),
                                      adj_final_value).to(CFG.finetune_device)

    # adj_new = torch.index_select(adj,1,torch.tensor(idx_pos).to(CFG.finetune_device))

    features_append = deepcopy(features[idx_add, :])
    labels_append = deepcopy(labels[idx_add])
    # for i in range(range(num_upsample)):
    #     idx_pos += idx_pos
    idx_new = torch.tensor([i for i in range(adj.shape[0],num_node)]).to(CFG.finetune_device)
    feature_before = features[:labels.shape[0],:]
    feature_after = features[labels.shape[0]:,:]
    label_fill = [0 for i in range(len(feature_after))]

    feature_combine = torch.cat((feature_before,features_append,feature_after),dim=0).to(CFG.finetune_device)
    # features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels,torch.tensor(label_fill).to(CFG.finetune_device),labels_append), 0).to(CFG.finetune_device)
    idx_train = torch.cat((idx_train, idx_new), 0).to(CFG.finetune_device)

    return adj_new, feature_combine, labels, idx_train


# src_upsample(adj, features, labels, idx_train, portion=1.0, im_class_num=1)



class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self,  alpha=None,class_num=30, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(10000, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        self.alpha = self.alpha.to(CFG.finetune_device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
