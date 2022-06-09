from sklearn.metrics import  f1_score
import numpy as np
import random
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
import math

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def F1(output, labels):
    output = output.argmax(1)
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, output,average='macro')
    return micro


def get_performance(logits,y):
    TP,TN,FN,FP = [],[],[],[]
    TP.append(((logits.argmax(1) == 1) & (y.squeeze() == 1)).sum().item())
    TN.append(((logits.argmax(1) == 0) & (y.squeeze() == 0)).sum().item())
    FN.append(((logits.argmax(1) == 0) & (y.squeeze() == 1)).sum().item())
    FP.append(((logits.argmax(1) == 1) & (y.squeeze() == 0)).sum().item())


    TP_ave = np.mean(TP)
    TN_ave =np.mean(TN)
    FN_ave = np.mean(FN)
    FP_ave = np.mean(FP)

    p = TP_ave / (TP_ave + FP_ave)
    r = TP_ave / (TP_ave + FN_ave)
    F1 = 2 * r * p / (r + p)
    acc = (TP_ave + TN_ave) / (TP_ave + TN_ave + FP_ave + FN_ave)

    return F1,acc,r,p

def get_balancenessed(logits,y):
    TP,TN,FN,FP = [],[],[],[]
    TP.append(((logits.argmax(1) == 1) & (y.squeeze() == 1)).sum().item())
    TN.append(((logits.argmax(1) == 0) & (y.squeeze() == 0)).sum().item())
    FN.append(((logits.argmax(1) == 0) & (y.squeeze() == 1)).sum().item())
    FP.append(((logits.argmax(1) == 1) & (y.squeeze() == 0)).sum().item())

    acc_1 = TP[0] / (y.squeeze() == 1).sum().item()
    acc_0 = TN[0] / (y.squeeze() == 0).sum().item()
    b = math.exp(-(acc_0 - acc_1)**2 / 0.01) /2

    return acc_0,acc_1,b



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_performance_lr(logits,y):
    TP,TN,FN,FP = [],[],[],[]
    TP.append(((logits == 1) & (y == 1)).sum().item())
    TN.append(((logits == 0) & (y == 0)).sum().item())
    FN.append(((logits == 0) & (y == 1)).sum().item())
    FP.append(((logits == 1) & (y == 0)).sum().item())


    TP_ave = np.mean(TP)
    TN_ave =np.mean(TN)
    FN_ave = np.mean(FN)
    FP_ave = np.mean(FP)

    p = TP_ave / (TP_ave + FP_ave)
    r = TP_ave / (TP_ave + FN_ave)
    F1 = 2 * r * p / (r + p)
    acc = (TP_ave + TN_ave) / (TP_ave + TN_ave + FP_ave + FN_ave)

    return F1,acc,r,p


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_lord_error_fn(logits, Y, ord):
  errors = torch.nn.functional.softmax(logits,dim=1) - Y
  scores = np.linalg.norm(errors.detach().cpu().numpy(), ord=ord, axis=-1)

  return scores


def get_label(feature):

    print("calculating the similairty")
    sims = cosine_similarity(feature.cpu().detach().numpy())
    print("finishing the similairty calculation")

    k = 3

    fo = open('./data/github_data/intra_label_3.txt','w',encoding='utf-8')

    sort_index = np.argsort(sims, axis=1)
    for line in range(sims.shape[0]):
        for col in range(2 * k):
            if col < k:
                fo.write("\t".join([str(line),str(int(sort_index[line, col])),str(0)])+'\n')
            else:
                fo.write("\t".join([str(line), str(int(sort_index[line, -col + 1])), str(1)]) + '\n')

    print("the end of get_label")