import cv2
from matplotlib import pyplot as plt
import albumentations as A
import numpy as np
import scipy.sparse as sp
import torch
import config as CFG
import random



class NodeImageDataset(torch.utils.data.Dataset):
    def __init__(self, entity_id,image,label, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = list(image)
        # self.captions = list(captions)
        self.entity_id = entity_id
        self.transforms = transforms
        self.label = label



    def __getitem__(self, idx):

        item = {}


        image = plt.imread(self.image_filenames[idx])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['entity'] = self.entity_id[idx]
        item['label'] = self.label[idx]

        return item


    def __len__(self):
        return len(self.image_filenames)


class NodeTextDataset(torch.utils.data.Dataset):
    def __init__(self, entity_id,text,label,tokenizer):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.text = list(text)

        self.encoded_text = tokenizer(
            self.text, padding=True, truncation=True, max_length=CFG.max_length
        )
        self.entity_id = entity_id
        # self.transforms = transforms
        self.label = label



    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_text.items()
        }

        item['entity'] = self.entity_id[idx]
        item['label'] = self.label[idx]

        return item


    def __len__(self):
        return len(self.text)


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(    # 组合变换
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),  # resize 到同一个size
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )



def load_data( dataset="darknet"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    if CFG.data == 'insta':
        embed_dir = CFG.insta_feature_path
        relation_dir = CFG.insta_relation_path
        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
        features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)
        number = 8225
        label = encode_onehot(idx_features_labels[:number, 1])

        # build graph
        # idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
        idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    elif CFG.data == 'github':

        embed_dir = CFG.github_feature_path
        label_dir = CFG.github_label_path
        relation_dir = CFG.githu_relation_path

        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=',', invalid_raise=True)
        features = sp.csr_matrix(idx_features_labels[:, :], dtype=np.float32)

        label_data = np.genfromtxt(label_dir,
                                   dtype=np.dtype(int), delimiter='\t', invalid_raise=True)
        label = encode_onehot(label_data[:, -1])

        # build graph
        idx = np.array(list(range(idx_features_labels.shape[0])),dtype=np.int32)
        # idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}


    edges_unordered = np.genfromtxt(relation_dir,
                                    dtype=np.int32)[:,:-1]

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    index = [i for i in range(label.shape[0])]
    length = len(index)
    random.shuffle(index)


    idx_train = index[:int(length * 0.5)]
    idx_val = index[int(length * 0.5):int(length * 0.75)]
    idx_test = index[int(length * 0.75):]


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    features = torch.FloatTensor(np.array(features.todense())).to(CFG.finetune_device)
    labels = torch.LongTensor(np.where(label)[1]).to(CFG.finetune_device)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(CFG.finetune_device)

    idx_train = torch.LongTensor(idx_train).to(CFG.finetune_device)
    idx_val = torch.LongTensor(idx_val).to(CFG.finetune_device)
    idx_test = torch.LongTensor(idx_test).to(CFG.finetune_device)

    return adj, features, labels, idx_train, idx_val, idx_test

def split_genuine(labels):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/10 *8)
            c_num_mat[i,1] = int(c_num/10 * 1)
            c_num_mat[i,2] = int(c_num/10 * 1)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx).to(CFG.finetune_device)
    val_idx = torch.LongTensor(val_idx).to(CFG.finetune_device)
    test_idx = torch.LongTensor(test_idx).to(CFG.finetune_device)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat

def load_finetune_data():

    if CFG.data == 'insta':
        embed_dir = CFG.insta_feature_path
        relation_dir = CFG.insta_relation_path
        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
        features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)
        number = 8225
        label = encode_onehot(idx_features_labels[:number, 1])

        # build graph
        # idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
        idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    elif CFG.data == 'github':

        embed_dir = r'D:\Muco\email2\all_embedding_bert128_model-1.txt'
        label_dir = r"D:\Muco\email2\malware_label.txt"
        relation_dir = r"D:\Muco\email2\all_rel_without_gt.txt"

        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=',', invalid_raise=True)

        # idx_features_labels[:,:embed.shape[0]] = embed

        features = sp.csr_matrix(idx_features_labels[:, :], dtype=np.float32)

        label_data = np.genfromtxt(label_dir,
                                   dtype=np.dtype(int), delimiter='\t', invalid_raise=True)[:, -1]
        label = encode_onehot(label_data)

        # build graph
        idx = np.array(list(range(idx_features_labels.shape[0])),dtype=np.int32)
        # idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    elif CFG.data == 'yelp':

        fea_dir = CFG.yelp_feature_path
        relation_dir = CFG.yelp_relation_path
        idx_features_labels = np.genfromtxt(fea_dir,
                                            dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
        features = idx_features_labels[:, 2:]
        if CFG.yelp_concate == 1.0:
            embed_dir = CFG.yelp_embed_path
            idx_embeds_labels = np.genfromtxt(embed_dir,
                                              dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
            embeds = idx_embeds_labels[:, 2:]

            features = np.concatenate((features,embeds),axis=1)

        features = sp.csr_matrix(features, dtype=np.float32)
        number = 67395
        label = torch.tensor(encode_onehot(idx_features_labels[:number, 1]))


        # build graph
        # idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
        idx = np.array([i for i in range(features.shape[0])])
        # idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    # relation_dir = r"D:\Crossmodal\data\github_data\all_rel_id.txt"

    edges_unordered = np.genfromtxt(relation_dir,
                                    dtype=np.int32)[:,:-1]

    # edges_unordered = np.genfromtxt(r"D:\Muco\data\for embedding\remove_test_all.relation",
    #                                 dtype=np.int32)

    # edges_unordered = np.genfromtxt(r"E:\code\Muco\data\for embedding\all.relation",
    #                                 dtype=np.int32)

    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # if CFG.node_backbone != 'sage':
    #     adj = normalize(adj + sp.eye(adj.shape[0]))
    # else:
    #     adj = adj + sp.eye(adj.shape[0])

    # idx_train, idx_val, idx_test, c_num_mat = split_genuine(torch.tensor(label_data))


    
    index = [i for i in range(label.shape[0])]
    length = len(index)
    random.shuffle(index)


    # idx_train = index[:int(length * 0.8)]
    # idx_val = index[int(length * 0.8):int(length * 0.9)]
    # idx_test = index[int(length * 0.9):]

    # idx_train = index[:int(length * 0.7)]
    # idx_val = index[int(length * 0.7):int(length * 0.8)]
    # idx_test = index[int(length * 0.8):]

    # idx_train = index[:int(length * 0.5)]
    # idx_val = index[int(length * 0.5):int(length * 0.6)]
    # idx_test = index[int(length * 0.6):]


    idx_train = index[:int(length * 0.2)]
    idx_val = index[int(length * 0.2):int(length * 0.3)]
    idx_test = index[int(length * 0.3):]


    # idx_train = index[:int(length * 0.5)]
    # idx_val = index[int(length * 0.5):int(length * 0.75)]
    # idx_test = index[int(length * 0.75):]

    # idx_train = index[:int(length * 0.5)]
    # idx_val = index[int(length * 0.5):int(length * 0.75)]
    # idx_test = index[int(length * 0.75):]

    # idx_train = range(0,8650,3)
    # idx_val = range(1,8650,3)
    # idx_test = range(2,8650,3)

    '''
    neg_index = [i for i, j in enumerate(label) if j[0] == 1]
    pos_index = [i for i, j in enumerate(label) if j[0] == 0]

    # idx = [i for i in range(8651)]

    # idx_train = range(0, labels.shape[0], 3)
    # idx_val = range(1, labels.shape[0], 3)
    # idx_test = range(2, labels.shape[0], 3)

    random.shuffle(neg_index)
    random.shuffle(pos_index)

    # train_ind = pos_index[:int(len(pos_index) * 0.5)] + neg_index[:10 * int(len(pos_index) * 0.5)]
    # val_ind = pos_index[int(len(pos_index) * 0.5):int(len(pos_index) * 0.75)] + \
    #           neg_index[10 * int(len(pos_index) * 0.5): 10 * int(len(pos_index) * 0.5) + int(len(pos_index) * 0.25)]
    # test_ind = pos_index[int(len(pos_index) * 0.75):int(len(pos_index) * 1.0)] + \
    #            neg_index[
    #            10 * int(len(pos_index) * 0.5) + int(len(pos_index) * 0.25):10 * int(len(pos_index) * 0.5) + int(
    #                len(pos_index) * 0.5)]


    # idx_train = pos_index[:int(len(pos_index) * 0.7)] + neg_index[:8*int(len(pos_index) * 0.7)]
    # idx_val = pos_index[int(len(pos_index) * 0.7):int(len(pos_index) * 0.8)] + \
    #           neg_index[8*int(len(pos_index) * 0.7): 8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.1)]
    # idx_test = pos_index[int(len(pos_index) * 0.8):int(len(pos_index) * 1.0)] + \
    #            neg_index[8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.1):8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.3)]


    idx_train = pos_index[:int(len(pos_index) * 0.5)] + neg_index[:10 * int(len(pos_index) * 0.5)]
    idx_val = pos_index[int(len(pos_index) * 0.5):int(len(pos_index) * 0.6)] + \
              neg_index[10 * int(len(pos_index) * 0.5): 10 * int(len(pos_index) * 0.5) + int(len(pos_index) * 0.1)]
    idx_test = pos_index[int(len(pos_index) * 0.6):int(len(pos_index) * 1.0)] + \
               neg_index[10 * int(len(pos_index) * 0.5) + int(len(pos_index) * 0.1):10 * int(len(pos_index) * 0.5) + int(
                   len(pos_index) * 0.5)]

    # idx_train = pos_index[:int(len(pos_index) * 0.2)] + neg_index[:10 * int(len(pos_index) * 0.2)]
    # idx_val = pos_index[int(len(pos_index) * 0.2):int(len(pos_index) * 0.3)] + \
    #           neg_index[10 * int(len(pos_index) * 0.2): 10 * int(len(pos_index) * 0.2) + int(len(pos_index) * 0.1)]
    # idx_test = pos_index[int(len(pos_index) * 0.3):int(len(pos_index) * 1.0)] + \
    #            neg_index[10 * int(len(pos_index) * 0.2) + int(len(pos_index) * 0.1):10 * int(len(pos_index) * 0.2) + int(
    #                len(pos_index) * 0.8)]

    # idx_train = pos_index[:int(len(pos_index) * 0.2)] + neg_index[:int(len(pos_index) * 0.2)]
    # idx_val = pos_index[int(len(pos_index) * 0.2):int(len(pos_index) * 0.3)] + neg_index[int(len(pos_index) * 0.2):int(
    #     len(pos_index) * 0.3)]
    # idx_test = pos_index[int(len(pos_index) * 0.3):int(len(pos_index) * 1.0)] + neg_index[int(len(pos_index) * 0.3):int(
    #     len(pos_index) * 1.0)]

    '''

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_train = range(0, labels.shape[0], 3)
    # idx_val = range(1, labels.shape[0], 3)
    # idx_test = range(2, labels.shape[0], 3)

    # idx_train = range(0,idx_features_labels.shape[0],3)
    # idx_val = range(1,idx_features_labels.shape[0],3)
    # idx_test = range(2,idx_features_labels.shape[0],3)


    # idx_pair = pair_labels[:,:-1]
    #
    # idx_train = range(0,pair_labels.shape[0],3)
    # idx_val = range(1,pair_labels.shape[0],3)
    # idx_test = range(2,pair_labels.shape[0],3)

    idx_train = torch.LongTensor(idx_train).to(CFG.finetune_device)
    idx_val = torch.LongTensor(idx_val).to(CFG.finetune_device)
    idx_test = torch.LongTensor(idx_test).to(CFG.finetune_device)
    

    features = torch.FloatTensor(np.array(features.todense())).to(CFG.finetune_device)
    labels = torch.LongTensor(np.where(label)[1]).to(CFG.finetune_device)
    # labels_a = torch.LongTensor(labels).to(CFG.finetune_device)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(CFG.finetune_device)



    # return adj, features, labels, idx_train, idx_val, idx_test,idx_pair
    # return adj, features, labels,label, idx_train, idx_val, idx_test,c_num_mat
    return adj, features, labels,label, idx_train, idx_val, idx_test


def load_finetune_data_for_imbalanced():
    if CFG.data == 'insta':
        embed_dir = CFG.insta_feature_path
        relation_dir = CFG.insta_relation_path
        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=' ', invalid_raise=True)

        features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)

        # features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)
        number = 8225
        label = encode_onehot(idx_features_labels[:number, 1])

        # build graph
        # idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
        idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    elif CFG.data == 'github':

        embed_dir = r'D:\Muco\email2\all_embedding_bert128_model-1.txt'
        label_dir = r"D:\Muco\email2\malware_label.txt"
        relation_dir = r"D:\Muco\email2\all_rel_without_gt.txt"

        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=',', invalid_raise=True)

        # idx_features_labels[:,:embed.shape[0]] = embed

        features = sp.csr_matrix(idx_features_labels[:, :], dtype=np.float32)

        label_data = np.genfromtxt(label_dir,
                                   dtype=np.dtype(int), delimiter='\t', invalid_raise=True)[:, -1]
        label = encode_onehot(label_data)

        # build graph
        idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
        # idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    elif CFG.data == 'yelp':

        fea_dir = CFG.yelp_feature_path
        relation_dir = CFG.yelp_relation_path
        idx_features_labels = np.genfromtxt(fea_dir,
                                            dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
        features = idx_features_labels[:, 2:]
        if CFG.yelp_concate == 1.0:
            embed_dir = CFG.yelp_embed_path
            idx_embeds_labels = np.genfromtxt(embed_dir,
                                              dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
            embeds = idx_embeds_labels[:, 2:]

            features = np.concatenate((features, embeds), axis=1)

        features = sp.csr_matrix(features, dtype=np.float32)
        number = 67395
        label = torch.tensor(encode_onehot(idx_features_labels[:number, 1]))

        # build graph
        # idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
        idx = np.array([i for i in range(features.shape[0])])
        # idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    # relation_dir = r"D:\Crossmodal\data\github_data\all_rel_id.txt"

    edges_unordered = np.genfromtxt(relation_dir,
                                    dtype=np.int32)[:, :-1]

    # edges_unordered = np.genfromtxt(r"D:\Muco\data\for embedding\remove_test_all.relation",
    #                                 dtype=np.int32)

    # edges_unordered = np.genfromtxt(r"E:\code\Muco\data\for embedding\all.relation",
    #                                 dtype=np.int32)

    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # if CFG.node_backbone != 'sage':
    #     adj = normalize(adj + sp.eye(adj.shape[0]))
    # else:
    #     adj = adj + sp.eye(adj.shape[0])

    # idx_train, idx_val, idx_test, c_num_mat = split_genuine(torch.tensor(label_data))

    index = [i for i in range(label.shape[0])]
    length = len(index)
    random.shuffle(index)

    neg_index = [i for i, j in enumerate(label) if j[0] == 1]
    pos_index = [i for i, j in enumerate(label) if j[0] == 0]

    random.shuffle(neg_index)
    random.shuffle(pos_index)

    print(len(pos_index))
    print(len(neg_index))

    if len(pos_index)< len(neg_index):
        min_index = pos_index
        max_index = neg_index
    else:
        max_index = pos_index
        min_index = neg_index

    # train_ind = pos_index[:int(len(pos_index) * 0.7)] + neg_index[:8*int(len(pos_index) * 0.7)]
    # val_ind = pos_index[int(len(pos_index) * 0.7):int(len(pos_index) * 0.8)] + \
    #           neg_index[8*int(len(pos_index) * 0.7): 8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.1)]
    # test_ind = pos_index[int(len(pos_index) * 0.8):int(len(pos_index) * 1.0)] + \
    #            neg_index[8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.1):8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.3)]

    # 1:10
    train_ind = min_index[:1000] + max_index[:10000]
    val_ind = min_index[1000:1500] + \
              max_index[10000:15000]
    test_ind = min_index[1500:2000] + \
               max_index[15000:20000]

    #1:100   3341  20676
    # train_ind = pos_index[:100] + neg_index[:10000]
    # val_ind = pos_index[100:150] + \
    #           neg_index[10000:15000]
    # test_ind = pos_index[150:200] + \
    #            neg_index[15000:20000]

    # orig ratio
    # train_ind = range(0, label.shape[0], 3)
    # val_ind = range(1, label.shape[0], 3)
    # test_ind = range(2, label.shape[0], 3)

    # 1:10   # 2985   # 5240
    # train_ind = pos_index[:300] + neg_index[:3000]
    # val_ind = pos_index[300:400] + \
    #           neg_index[3000:4000]
    # test_ind = pos_index[400:550] + \
    #            neg_index[4000:5200]

    # 1:100  # 2985   # 5240
    # train_ind = pos_index[:30] + neg_index[:3000]
    # val_ind = pos_index[30:40] + \
    #           neg_index[3000:4000]
    # test_ind = pos_index[40:55] + \
    #            neg_index[4000:5200]



    # 1:50 # 2985   # 5240
    # train_ind = neg_index[:60] + pos_index[:3000]
    # val_ind = neg_index[60:80] + \
    #           pos_index[3000:4000]
    # test_ind = neg_index[80:120] + \
    #            pos_index[4000:5200]

    # train_ind = min_index[:60] + max_index[:3000]
    # val_ind = min_index[60:80] + \
    #           max_index[3000:4000]
    # test_ind = min_index[80:120] + \
    #            max_index[4000:5200]

    # 1:100 # 2985   # 5240
    # train_ind = min_index[:30] + max_index[:3000]
    # val_ind = min_index[30:40] + \
    #           max_index[3000:4000]
    # test_ind = min_index[40:55] + \
    #            max_index[4000:5200]

    idx_train = torch.LongTensor(train_ind)
    idx_val = torch.LongTensor(val_ind)
    idx_test = torch.LongTensor(test_ind)



    # idx_train = range(0,idx_features_labels.shape[0],3)
    # idx_val = range(1,idx_features_labels.shape[0],3)
    # idx_test = range(2,idx_features_labels.shape[0],3)

    # idx_pair = pair_labels[:,:-1]
    #
    # idx_train = range(0,pair_labels.shape[0],3)
    # idx_val = range(1,pair_labels.shape[0],3)
    # idx_test = range(2,pair_labels.shape[0],3)

    idx_train = torch.LongTensor(idx_train).to(CFG.finetune_device)
    idx_val = torch.LongTensor(idx_val).to(CFG.finetune_device)
    idx_test = torch.LongTensor(idx_test).to(CFG.finetune_device)

    features = torch.FloatTensor(np.array(features.todense())).to(CFG.finetune_device)
    labels = torch.LongTensor(np.where(label)[1]).to(CFG.finetune_device)
    # labels_a = torch.LongTensor(labels).to(CFG.finetune_device)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(CFG.finetune_device)

    # return adj, features, labels, idx_train, idx_val, idx_test,idx_pair
    # return adj, features, labels,label, idx_train, idx_val, idx_test,c_num_mat
    return adj, features, labels, idx_train, idx_val, idx_test

def load_finetune_data_for_imbalanced_github():


    embed_dir = r'D:\Muco\email2\all_embedding_bert128_model-1.txt'
    label_dir = r"D:\Muco\email2\malware_label.txt"
    relation_dir = r"D:\Muco\email2\all_rel_without_gt.txt"

    idx_features_labels = np.genfromtxt(embed_dir,
                                        dtype=np.dtype(str), delimiter=',', invalid_raise=True)

    # idx_features_labels[:,:embed.shape[0]] = embed

    features = sp.csr_matrix(idx_features_labels[:, :], dtype=np.float32)

    label_data = np.genfromtxt(label_dir,
                               dtype=np.dtype(int), delimiter='\t', invalid_raise=True)[:, -1]
    label = encode_onehot(label_data)

    # build graph
    idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
    # idx = np.array(idx_features_labels[:, 0], dtype=np.float)
    idx_map = {j: i for i, j in enumerate(idx)}


    # relation_dir = r"D:\Crossmodal\data\github_data\all_rel_id.txt"

    edges_unordered = np.genfromtxt(relation_dir,
                                    dtype=np.int32)[:, :-1]

    # edges_unordered = np.genfromtxt(r"D:\Muco\data\for embedding\remove_test_all.relation",
    #                                 dtype=np.int32)

    # edges_unordered = np.genfromtxt(r"E:\code\Muco\data\for embedding\all.relation",
    #                                 dtype=np.int32)

    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # if CFG.node_backbone != 'sage':
    #     adj = normalize(adj + sp.eye(adj.shape[0]))
    # else:
    #     adj = adj + sp.eye(adj.shape[0])

    # idx_train, idx_val, idx_test, c_num_mat = split_genuine(torch.tensor(label_data))

    index = [i for i in range(label.shape[0])]
    length = len(index)
    random.shuffle(index)

    neg_index = [i for i, j in enumerate(label) if j[0] == 1]
    pos_index = [i for i, j in enumerate(label) if j[0] == 0]

    random.shuffle(neg_index)
    random.shuffle(pos_index)

    print(len(pos_index))
    print(len(neg_index))

    if len(pos_index)< len(neg_index):
        min_index = pos_index
        max_index = neg_index
    else:
        max_index = pos_index
        min_index = neg_index

    # train_ind = pos_index[:int(len(pos_index) * 0.7)] + neg_index[:8*int(len(pos_index) * 0.7)]
    # val_ind = pos_index[int(len(pos_index) * 0.7):int(len(pos_index) * 0.8)] + \
    #           neg_index[8*int(len(pos_index) * 0.7): 8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.1)]
    # test_ind = pos_index[int(len(pos_index) * 0.8):int(len(pos_index) * 1.0)] + \
    #            neg_index[8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.1):8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.3)]

    # 1:10
    train_ind = min_index[:1000] + max_index[:10000]
    val_ind = min_index[1000:1500] + \
              max_index[10000:15000]
    test_ind = min_index[1500:2000] + \
               max_index[15000:20000]

    #1:100   3341  20676
    # train_ind = pos_index[:100] + neg_index[:10000]
    # val_ind = pos_index[100:150] + \
    #           neg_index[10000:15000]
    # test_ind = pos_index[150:200] + \
    #            neg_index[15000:20000]

    # orig ratio
    # train_ind = range(0, label.shape[0], 3)
    # val_ind = range(1, label.shape[0], 3)
    # test_ind = range(2, label.shape[0], 3)

    # 1:10   # 2985   # 5240
    # train_ind = pos_index[:300] + neg_index[:3000]
    # val_ind = pos_index[300:400] + \
    #           neg_index[3000:4000]
    # test_ind = pos_index[400:550] + \
    #            neg_index[4000:5200]

    # 1:100  # 2985   # 5240
    # train_ind = pos_index[:30] + neg_index[:3000]
    # val_ind = pos_index[30:40] + \
    #           neg_index[3000:4000]
    # test_ind = pos_index[40:55] + \
    #            neg_index[4000:5200]



    # 1:50 # 2985   # 5240
    # train_ind = neg_index[:60] + pos_index[:3000]
    # val_ind = neg_index[60:80] + \
    #           pos_index[3000:4000]
    # test_ind = neg_index[80:120] + \
    #            pos_index[4000:5200]

    # train_ind = min_index[:60] + max_index[:3000]
    # val_ind = min_index[60:80] + \
    #           max_index[3000:4000]
    # test_ind = min_index[80:120] + \
    #            max_index[4000:5200]

    # 1:100 # 2985   # 5240
    # train_ind = min_index[:30] + max_index[:3000]
    # val_ind = min_index[30:40] + \
    #           max_index[3000:4000]
    # test_ind = min_index[40:55] + \
    #            max_index[4000:5200]

    idx_train = torch.LongTensor(train_ind)
    idx_val = torch.LongTensor(val_ind)
    idx_test = torch.LongTensor(test_ind)



    # idx_train = range(0,idx_features_labels.shape[0],3)
    # idx_val = range(1,idx_features_labels.shape[0],3)
    # idx_test = range(2,idx_features_labels.shape[0],3)

    # idx_pair = pair_labels[:,:-1]
    #
    # idx_train = range(0,pair_labels.shape[0],3)
    # idx_val = range(1,pair_labels.shape[0],3)
    # idx_test = range(2,pair_labels.shape[0],3)

    idx_train = torch.LongTensor(idx_train).to(CFG.finetune_device)
    idx_val = torch.LongTensor(idx_val).to(CFG.finetune_device)
    idx_test = torch.LongTensor(idx_test).to(CFG.finetune_device)

    features = torch.FloatTensor(np.array(features.todense())).to(CFG.finetune_device)
    labels = torch.LongTensor(np.where(label)[1]).to(CFG.finetune_device)
    # labels_a = torch.LongTensor(labels).to(CFG.finetune_device)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(CFG.finetune_device)

    # return adj, features, labels, idx_train, idx_val, idx_test,idx_pair
    # return adj, features, labels,label, idx_train, idx_val, idx_test,c_num_mat
    return adj, features, labels, idx_train, idx_val, idx_test

def load_finetune_data_for_imbalanced_insta():

    embed_dir = CFG.insta_feature_path
    relation_dir = CFG.insta_relation_path
    idx_features_labels = np.genfromtxt(embed_dir,
                                        dtype=np.dtype(str), delimiter=' ', invalid_raise=True)

    features = sp.csr_matrix(idx_features_labels[:, 2:770], dtype=np.float32)

    # features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)
    number = 8225
    label = encode_onehot(idx_features_labels[:number, 1])

    # build graph
    # idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
    idx = np.array(idx_features_labels[:, 0], dtype=np.float)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt(relation_dir,
                                    dtype=np.int32)[:, :-1]

    # edges_unordered = np.genfromtxt(r"D:\Muco\data\for embedding\remove_test_all.relation",
    #                                 dtype=np.int32)

    # edges_unordered = np.genfromtxt(r"E:\code\Muco\data\for embedding\all.relation",
    #                                 dtype=np.int32)

    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # if CFG.node_backbone != 'sage':
    #     adj = normalize(adj + sp.eye(adj.shape[0]))
    # else:
    #     adj = adj + sp.eye(adj.shape[0])

    # idx_train, idx_val, idx_test, c_num_mat = split_genuine(torch.tensor(label_data))

    index = [i for i in range(label.shape[0])]
    length = len(index)
    random.shuffle(index)

    neg_index = [i for i, j in enumerate(label) if j[0] == 1]
    pos_index = [i for i, j in enumerate(label) if j[0] == 0]

    random.shuffle(neg_index)
    random.shuffle(pos_index)

    # train_ind = pos_index[:int(len(pos_index) * 0.7)] + neg_index[:8*int(len(pos_index) * 0.7)]
    # val_ind = pos_index[int(len(pos_index) * 0.7):int(len(pos_index) * 0.8)] + \
    #           neg_index[8*int(len(pos_index) * 0.7): 8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.1)]
    # test_ind = pos_index[int(len(pos_index) * 0.8):int(len(pos_index) * 1.0)] + \
    #            neg_index[8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.1):8*int(len(pos_index) * 0.7)+ int(len(pos_index) * 0.3)]

    # 1:10
    # train_ind = pos_index[:1000] + neg_index[:10000]
    # val_ind = pos_index[1000:1500] + \
    #           neg_index[10000:15000]
    # test_ind = pos_index[1500:2000] + \
    #            neg_index[15000:20000]

    #1:100   3341  20676
    # train_ind = pos_index[:100] + neg_index[:10000]
    # val_ind = pos_index[100:150] + \
    #           neg_index[10000:15000]
    # test_ind = pos_index[150:200] + \
    #            neg_index[15000:20000]

    # orig ratio
    idx_train = range(0, label.shape[0], 3)
    idx_val = range(1, label.shape[0], 3)
    idx_test = range(2, label.shape[0], 3)

    # 1:10   # 2985   # 5240
    # train_ind = pos_index[:300] + neg_index[:3000]
    # val_ind = pos_index[300:400] + \
    #           neg_index[3000:4000]
    # test_ind = pos_index[400:550] + \
    #            neg_index[4000:5200]

    # 1:100  # 2985   # 5240
    # train_ind = pos_index[:30] + neg_index[:3000]
    # val_ind = pos_index[30:40] + \
    #           neg_index[3000:4000]
    # test_ind = pos_index[40:55] + \
    #            neg_index[4000:5200]

    print(len(pos_index))
    print(len(neg_index))

    if len(pos_index)< len(neg_index):
        min_index = pos_index
        max_index = neg_index
    else:
        max_index = pos_index
        min_index = neg_index

    # 1:50 # 2985   # 5240
    # train_ind = neg_index[:60] + pos_index[:3000]
    # val_ind = neg_index[60:80] + \
    #           pos_index[3000:4000]
    # test_ind = neg_index[80:120] + \
    #            pos_index[4000:5200]

    # train_ind = min_index[:60] + max_index[:3000]
    # val_ind = min_index[60:80] + \
    #           max_index[3000:4000]
    # test_ind = min_index[80:120] + \
    #            max_index[4000:5200]

    # 1:100 # 2985   # 5240
    # idx_train = min_index[:30] + max_index[:3000]
    # idx_val = min_index[30:40] + \
    #           max_index[3000:4000]
    # idx_test = min_index[40:55] + \
    #            max_index[4000:5200]


    # 1:50 # 2985   # 5240
    # idx_train = min_index[:60] + max_index[:3000]
    # idx_val = min_index[60:80] + \
    #           max_index[3000:4000]
    # idx_test = min_index[80:120] + \
    #            max_index[4000:5200]

    #
    # idx_train = torch.LongTensor(train_ind)
    # idx_val = torch.LongTensor(val_ind)
    # idx_test = torch.LongTensor(test_ind)


    # 1:10   # 2985   # 5240
    # idx_train = min_index[:300] + max_index[:3000]
    # idx_val = min_index[300:400] + \
    #           max_index[3000:4000]
    # idx_test = min_index[400:550] + \
    #            max_index[4000:5200]




    # idx_train = range(0,idx_features_labels.shape[0],3)
    # idx_val = range(1,idx_features_labels.shape[0],3)
    # idx_test = range(2,idx_features_labels.shape[0],3)

    # idx_pair = pair_labels[:,:-1]
    #
    # idx_train = range(0,pair_labels.shape[0],3)
    # idx_val = range(1,pair_labels.shape[0],3)
    # idx_test = range(2,pair_labels.shape[0],3)

    idx_train = torch.LongTensor(idx_train).to(CFG.finetune_device)
    idx_val = torch.LongTensor(idx_val).to(CFG.finetune_device)
    idx_test = torch.LongTensor(idx_test).to(CFG.finetune_device)

    features = torch.FloatTensor(np.array(features.todense())).to(CFG.finetune_device)
    labels = torch.LongTensor(np.where(label)[1]).to(CFG.finetune_device)
    # labels_a = torch.LongTensor(labels).to(CFG.finetune_device)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(CFG.finetune_device)

    # return adj, features, labels, idx_train, idx_val, idx_test,idx_pair
    # return adj, features, labels,label, idx_train, idx_val, idx_test,c_num_mat
    return adj, features, labels, idx_train, idx_val, idx_test


def load_data_1(path="data/darknet/", dataset="darknet"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    if CFG.data == 'insta':
        idx_features_labels = np.genfromtxt(CFG.insta_feature_path,dtype=np.dtype(float), delimiter=' ', invalid_raise=True)
        relation_dir = CFG.insta_relation_path
        number = 8225
        # labels = encode_onehot(idx_features_labels[:number, 1])
    elif CFG.data == 'github':
        idx_features_labels = np.genfromtxt(CFG.github_feature_path,dtype=np.dtype(float), delimiter=' ', invalid_raise=True)
        relation_dir = CFG.github_relation_path
        number = 24017

    features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)

    labels = encode_onehot(idx_features_labels[:number, 1])



    edges_unordered = np.genfromtxt(relation_dir,
                                    dtype=np.int32)[:, :-1]

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}


    # relation_dir = CFG.relation_path
    #
    # edges_unordered = np.genfromtxt(relation_dir,
    #                                 dtype=np.int32)[:,:-1]


    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(idx_features_labels.shape[0], idx_features_labels.shape[0]),dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # idx_train,idx_val,idx_test = get_gcnlearndata(labels, neg_label = 0)
    # labels = encode_onehot(labels)

    # wrong_index_list = count_wrong_sample()

    ind = list(range(0,number))
    random.shuffle(ind)

    id_train = range(0,number,3)
    id_val = range(1,number,3)
    id_test = range(2,number,3)

    idx_train = [ind[i] for i in id_train]
    idx_val = [ind[j] for j in id_val]
    idx_test = [ind[k] for k in id_test]

    # idx_train = [ind[i] for i in id_train if ind[i] not in wrong_index_list]
    # idx_val = [ind[j] for j in id_val if ind[j] not in wrong_index_list]
    # idx_test = [ind[k] for k in id_test if ind[k] not in wrong_index_list]

    # idx_train = range(0,idx_features_labels.shape[0],3)
    # idx_val = range(1,idx_features_labels.shape[0],3)
    # idx_test = range(2,idx_features_labels.shape[0],3)

    features = torch.FloatTensor(np.array(features.todense())).to(CFG.device)
    labels = torch.LongTensor(np.where(labels)[1]).to(CFG.device)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(CFG.device)

    idx_train = torch.LongTensor(idx_train).to(CFG.device)
    idx_val = torch.LongTensor(idx_val).to(CFG.device)
    idx_test = torch.LongTensor(idx_test).to(CFG.device)

    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    # np.savetxt("cat.txt",labels_onehot,fmt="%s",delimiter=",")
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

from dgl.data.fakenews import FakeNewsDataset

# dataset = FakeNewsDataset('gossipcop', 'bert')
# graph, label = dataset[0]
# num_classes = dataset.num_classes
# feat = dataset.feature
# labels = dataset.labels
