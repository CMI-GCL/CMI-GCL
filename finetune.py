#!/usr/bin/env python
# encoding: utf-8



from modules import  NodeClassifier

from getdata import load_finetune_data_for_imbalanced_insta
from util import  get_performance
import torch.nn.functional as F
import torch.optim as optim
from main_graph_nlp_cl import *
import config as CFG
from util import seed_torch
from sklearn.metrics import roc_auc_score
import os
from torchgeometry.losses import FocalLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


seed_torch(42)
model = NodeClassifier().to(CFG.finetune_device)

optimizer = optim.Adam(model.parameters(),
                       lr=CFG.lr, weight_decay=CFG.weight_decay)

finetune = 1
epochs = 1000


pretrained_model_dir =r'.\pretrained\insta_node_image_cl_prune_0.1_temp_1.0_best_swintf_500_202205140727_both_feature.pt'


if finetune == 1:
    checkpoint = torch.load(pretrained_model_dir, map_location=CFG.finetune_device)

    model.nodencoder.gc1.weight.data = checkpoint['node_encoder.gc1.weight']
    model.nodencoder.gc1.bias.data = checkpoint['node_encoder.gc1.bias']
    model.nodencoder.gc2.weight.data = checkpoint['node_encoder.gc2.weight']
    model.nodencoder.gc2.bias.data = checkpoint['node_encoder.gc2.bias']


adj, features, labels, idx_train, idx_val, idx_test = load_finetune_data_for_imbalanced_insta()


model.train()
focaloss = FocalLoss(alpha=CFG.focal_alpha, gamma=CFG.focal_gamma, reduction='mean')

for epoch in range(epochs):
    optimizer.zero_grad()

    output = model(features, adj)
    logits = F.log_softmax(output, dim=1)

    loss_train = focaloss(output[idx_train], labels[idx_train])
    f1_train, acc_train, rec_train, prec_train = get_performance(logits[idx_train], labels[idx_train])
    auc_train = roc_auc_score(labels[idx_train].detach().cpu(),
                              F.softmax(output, dim=-1)[idx_train][:, 1].detach().cpu(), average='macro')

    loss_train.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()

        output = model(features, adj)
        logits = F.log_softmax(output, dim=1)

        loss_val = F.nll_loss(logits[idx_val], labels[idx_val])
        f1_val, acc_val, rec_val, prec_val = get_performance(logits[idx_val], labels[idx_val])


        auc_val = roc_auc_score(labels[idx_val].detach().cpu(), F.softmax(output, dim=-1)[idx_val][:, 1].detach().cpu(),
                                average='macro')
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train),
              'F1_train: {:.4f}'.format(f1_train),
              'auc_train: {:.4f}'.format(auc_train),
              'rec_train: {:.4f}'.format(rec_train),

              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val),
              'F1_val: {:.4f}'.format(f1_val),
              'auc_val: {:.4f}'.format(auc_val),
              'rec_val: {:.4f}'.format(rec_val),
              )


model.eval()

output = model(features, adj)
logits = F.log_softmax(output, dim=1)

loss_test = F.nll_loss(logits[idx_test], labels[idx_test])
f1_test, acc_test, rec_test, prec_test = get_performance(logits[idx_test], labels[idx_test])


auc_test = roc_auc_score(labels[idx_test].detach().cpu(), F.softmax(output,dim=-1)[idx_test][:,1].detach().cpu(), average='macro')

print('Epoch: {:04d}'.format(epoch + 1),
      'loss_test: {:.4f}'.format(loss_test.item()),
      'acc_test: {:.4f}'.format(acc_test),
      'F1_test: {:.4f}'.format(f1_test),
      'auc_test: {:.4f}'.format(auc_test),
      'rec_test: {:.4f}'.format(rec_test),
      )


