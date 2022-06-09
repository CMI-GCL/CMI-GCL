import torch

debug = True

data = 'insta'
# data = 'github'

number_github_rep = 24017
number_insta_user = 8225

node_image_matching_path = r".\data\insta_data\update\entityid_picid_label_duplicate.txt"
insta_feature_path = r'./data/insta_data/update/feature_duplicate.txt'
insta_relation_path = r".\data\insta_data\update\relation_id_duplicate.txt"

node_text_matching_path = r".\data\github_data\entityid_text_label.txt"
github_feature_path = r'.\data\github_data\all_feature.txt'
github_relation_path = r".\data\github_data\all_rel_id.txt"
github_label_path =  r".\data\github_data\malware_label.txt"

prune = True
finetune_prune = True

# prune_percent = 0.2
prune_percent = 0.1
finetune_prune_percent = 0.1

number_samples = 1000
# batch_size = 8
batch_size = 80
num_workers = 0
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
# epochs = 200
# epochs = 400
epochs = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

finetune_device = torch.device("cuda:1")
# finetune_device = torch.device("cuda")
# finetune_device = torch.device("cpu")

# image_encoder_model = 'resnet50'
image_encoder_model = 'swin_small_patch4_window7_224'
image_embedding = 768

# image_embedding = 2048
image_embedding = 768
# text_encoder_model = "distilbert-base-uncased"
# text_embedding = 768
# text_tokenizer = "distilbert-base-uncased"

text_encoder_model = "bert-base-uncased"
text_embedding = 768
text_tokenizer = "bert-base-uncased"

node_backbone = 'gcn'
# node_backbone = 'gat'
# node_backbone = 'sage'

imbalance_setting = 'reweight'   #['reweight','upsample','focal']
imbalance_up_scale = 10.0

upsample_portion = 10.0
##### gat #####
nheads = 8
alpha = 0.2
# focal_alpha = 5.0
focal_alpha = 1.0
focal_gamma = 2.0
max_length = 200

# insta_node_feature_dim = 1768
insta_node_feature_dim = 768
github_node_feature_dim = 128

node_feature_project_dim = 200
node_embedding_dim = 200

pretrained = False # for both image encoder and text encoder
trainable = False # for both image encoder and text encoder

temperature = 1.0
# temperature = 0.1
# temperature = 10
# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1


# node encoder
node_dropout = 0.5

feature_dim = 400
hidden_dim = 200
out_dim = 200
nclass = 2