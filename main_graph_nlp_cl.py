#!/usr/bin/env python
# encoding: utf-8

import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import DistilBertTokenizer
import config as CFG
from getdata import NodeTextDataset,load_finetune_data_for_imbalanced_github
from CMGCL import NodeTextCLModel
from util import AvgMeter, get_lr
import torch.nn.utils.prune as prune

from util import seed_torch,count_parameters

seed_torch(4)

def make_train_valid_dfs():
    dataframe = pd.read_csv(CFG.node_text_matching_path,sep='\t')  # image_id,caption,id
    max_id = dataframe.shape if not CFG.debug else 100
    text_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        text_ids, size=int(0.2 * len(text_ids)), replace=False
    )
    train_ids = [id_ for id_ in text_ids if id_ not in valid_ids]
    dataframe['id'] = list(dataframe.index)

    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)

    return train_dataframe, valid_dataframe

def make_repo_dfs():
    dataframe = pd.read_csv(CFG.node_text_matching_path,sep='	')  # image_id,caption,id
    max_id = dataframe.shape if not CFG.debug else dataframe.shape[0]
    text_ids = np.arange(0, max_id)

    repo_ids = [id_ for id_ in text_ids]
    dataframe['id'] = list(dataframe.index)
    repo_dataframe = dataframe[dataframe["id"].isin(repo_ids)].reset_index(drop=True)

    return repo_dataframe

def build_loaders(dataframe, tokenizer, mode):

    dataset = NodeTextDataset(
        dataframe["entity_id"].values,
        dataframe["text"].values,
        dataframe['label'].values,
        tokenizer=tokenizer
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def build_finetune_loaders(dataframe, tokenizer,mode):
    dataset = NodeTextDataset(
        dataframe["entity_id"].values,
        dataframe["text"].values,
        dataframe['label'].values,
        tokenizer=tokenizer,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataframe.shape[0],
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, feature,adj,train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss, node_embed_prune,node_embeds = model(batch, feature, adj)
        optimizer.zero_grad()
        if CFG.prune:
            for name, module in model.named_modules():
                if 'transformer' in name and isinstance(module, torch.nn.Linear):
                # if isinstance(module, torch.nn.Conv2d):
                    module.weight = module.weight_orig.clone()
                elif 'node_encoder_prune.' in name :
                    module.weight = module.weight_orig.clone()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["input_ids"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter,node_embed_prune,node_embeds


def valid_epoch(model, feature,adj,valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        # loss,node_embeds,scores,logits = model(batch,feature,adj)
        loss, node_embed_prune,node_embeds = model(batch, feature, adj)

        count = batch["input_ids"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter,node_embed_prune,node_embeds


def main():
    my_time = time.strftime('%Y%m%d%H%M', time.gmtime(time.time()))

    train_df, valid_df = make_train_valid_dfs()   # make train, validation datasets

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer) # read bert language model
    train_loader = build_loaders(train_df, tokenizer, mode="train")     #
    valid_loader = build_loaders(valid_df, tokenizer,mode="valid")

    model = NodeTextCLModel().to(CFG.device)
    count = count_parameters(model)
    print(CFG.node_backbone)
    print(CFG.text_encoder_model)
    print(count)


    if CFG.prune:
        for name, module in model.named_modules():
            if 'transformer' in name and isinstance(module,torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=int(module.weight.shape[0]*module.weight.shape[1]*CFG.prune_percent))
            elif 'node_encoder_prune.' in name:
                prune.l1_unstructured(module, name='weight', amount=int(module.weight.shape[0]*module.weight.shape[1]*CFG.prune_percent))
        print(dict(model.named_buffers()).keys())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    adj, features, labels, idx_train, idx_val, idx_test = load_finetune_data_for_imbalanced_github()

    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss,node_embed_prune_train,node_embeds_train = train_epoch(model,features,adj, train_loader, optimizer, lr_scheduler, step)

        model.eval()
        with torch.no_grad():
            valid_loss,node_embed_prune_val,node_embeds_val = valid_epoch(model,features,adj, valid_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "./pretrained/{}_node_text_cl_prune_{}_temp_{}_best_{}_{}.pt".format(CFG.data,CFG.prune_percent,CFG.temperature,CFG.epochs,my_time))
            print("Saved Best Model!")

    np.savetxt(r'./pretrained/{}_node_text_cl_prune_{}_temp_{}_best_{}_{}.txt'.format(CFG.data,CFG.prune_percent,CFG.temperature,CFG.epochs,my_time), node_embeds_train.cpu().data.numpy(),
               delimiter=',')
    np.savetxt(
        r'./pretrained/{}_node_text_cl_prune_{}_temp_{}_best_{}_{}_prune.txt'.format(CFG.data,CFG.prune_percent,CFG.temperature,CFG.epochs,my_time),
        node_embed_prune_train.cpu().data.numpy(),
        delimiter=',')

if __name__ == "__main__":
    main()
