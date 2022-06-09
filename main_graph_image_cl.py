#!/usr/bin/env python
# encoding: utf-8


import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import config as CFG
from getdata import NodeImageDataset, get_transforms,load_finetune_data_for_imbalanced
from CMGCL import NodeImageCLModel
from util import AvgMeter, get_lr,count_parameters
import torch.nn.utils.prune as prune
import time

def make_train_valid_dfs():
    dataframe = pd.read_csv(CFG.node_image_matching_path,sep='	')  # image_id,caption,id
    max_id = dataframe.shape if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    dataframe['id'] = list(dataframe.index)
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)

    return train_dataframe, valid_dataframe

def make_user_dfs():
    dataframe = pd.read_csv(CFG.node_image_matching_path,sep='	')  # image_id,caption,id
    max_id = dataframe.shape if not CFG.debug else 8224
    image_ids = np.arange(0, max_id)

    user_ids = [id_ for id_ in image_ids]
    dataframe['id'] = list(dataframe.index)
    user_dataframe = dataframe[dataframe["id"].isin(user_ids)].reset_index(drop=True)

    return user_dataframe

def build_loaders(dataframe, mode):
    transforms = get_transforms(mode=mode)
    dataset = NodeImageDataset(
        dataframe["entity_id"].values,
        dataframe["image_path"].values,
        dataframe['label'].values,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def build_finetune_loaders(dataframe, mode):
    transforms = get_transforms(mode=mode)
    dataset = NodeImageDataset(
        dataframe["entity_id"].values,
        dataframe["image_path"].values,
        dataframe['label'].values,
        # tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4112,
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
                if isinstance(module, torch.nn.Conv2d):
                    module.weight = module.weight_orig.clone()
                elif 'node_encoder_prune.' in name:
                    module.weight = module.weight_orig.clone()
        loss.backward()   # gradient
        optimizer.step()  # update weight

        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter,node_embed_prune,node_embeds


def valid_epoch(model, feature,adj,valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss, node_embed_prune,node_embeds = model(batch, feature, adj)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter,node_embed_prune,node_embeds

def main():
    my_time = time.strftime('%Y%m%d%H%M', time.gmtime(time.time()))
    train_df, valid_df = make_train_valid_dfs()   # make train, validation datasets

    train_loader = build_loaders(train_df, mode="train")     #
    valid_loader = build_loaders(valid_df, mode="valid")


    model = NodeImageCLModel().to(CFG.device)
    count = count_parameters(model)
    print(CFG.node_backbone)
    print(CFG.image_encoder_model)
    print(count)


    adj, features, labels, idx_train, idx_val, idx_test = load_finetune_data_for_imbalanced()
    if CFG.prune:
        for name, module in model.named_modules():
            if isinstance(module,torch.nn.Conv2d):
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

    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss,node_embed_prune_train,node_embeds_train = train_epoch(model,features,adj, train_loader, optimizer, lr_scheduler, step)


        model.eval()
        with torch.no_grad():
            valid_loss,node_embed_prune_val,node_embeds_val = valid_epoch(model,features,adj, valid_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "./pretrained/{}_node_image_cl_prune_{}_temp_{}_best_{}_{}_{}_both_feature.pt".format(CFG.data,CFG.prune_percent,CFG.temperature,CFG.image_encoder_model,CFG.epochs,my_time))
            print("Saved Best Model!")


    np.savetxt(r'./pretrained/{}_node_text_cl_prune_{}_temp_{}_best__{}_{}_{}.txt'.format(CFG.data,CFG.prune_percent,CFG.temperature,CFG.image_encoder_model,CFG.epochs,my_time), node_embeds_train.cpu().data.numpy(),
                   delimiter=',')
    np.savetxt(
            r'./pretrained/{}_node_text_cl_prune_{}_temp_{}_best_{}_{}_{}_prune.txt'.format(CFG.data,CFG.prune_percent,CFG.temperature,CFG.image_encoder_model,CFG.epochs,my_time),
            node_embed_prune_train.cpu().data.numpy(),delimiter=',')

if __name__ == "__main__":
    main()
