from torch import nn
import torch.nn.functional as F
import config as CFG
from modules import ImageEncoder, TextEncoder, NodeEncoder, ProjectionHead


class NodeImageCLModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        node_embedding=CFG.node_embedding_dim,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)

        self.node_encoder_prune = NodeEncoder()
        self.node_embed_prune_projection = ProjectionHead(embedding_dim=node_embedding)

        self.node_encoder = NodeEncoder()
        self.node_embed_projection = ProjectionHead(embedding_dim=node_embedding)
        self.temperature = temperature



    def forward(self, batch, feature, adj):

        # Getting Image and Node Features
        image_features = self.image_encoder(batch["image"])
        node_embed_prune = self.node_encoder_prune(feature.float(), adj)

        # Getting Text Embeddings
        image_embeddings = self.image_projection(image_features)
        node_embeddings_project_prune = self.node_embed_prune_projection(node_embed_prune)

        node_embedding_batch_prune = node_embeddings_project_prune[batch['entity']]

        # Calculating the inter-modality Loss
        logits_prune = (node_embedding_batch_prune @ image_embeddings.T) / self.temperature
        image_similarity = image_embeddings @ image_embeddings.T
        nodes_similarity_prune = node_embedding_batch_prune @ node_embedding_batch_prune.T
        targets = F.softmax(
            (image_similarity + nodes_similarity_prune) / 2 * self.temperature, dim=-1
        )
        nodes_loss = cross_entropy(logits_prune, targets, reduction='none')
        text_loss = cross_entropy(logits_prune.T, targets.T, reduction='none')
        loss = (text_loss + nodes_loss) / 2.0  # shape: (batch_size)

        # Getting the intra-modality Loss
        node_embed = self.node_encoder(feature.float(), adj)
        node_embeddings_project = self.node_embed_projection(node_embed)
        node_embedding_batch = node_embeddings_project[batch['entity']]

        logits_intra = (node_embedding_batch @ node_embedding_batch_prune.T) / self.temperature
        node_similarity_prune = node_embedding_batch_prune @ node_embedding_batch_prune.T

        nodes_similarity = node_embedding_batch @ node_embedding_batch.T

        targets_intra = F.softmax(
            (node_similarity_prune + nodes_similarity) / 2 * self.temperature, dim=-1
        )
        nodes_loss_intra = cross_entropy(logits_intra, targets_intra, reduction='none')
        node_loss_prune_intra = cross_entropy(logits_intra.T, targets_intra.T, reduction='none')
        loss_intra = (nodes_loss_intra + node_loss_prune_intra) / 2.0  # shape: (batch_size)


        loss = loss.mean() + loss_intra.mean()

        return loss, node_embed_prune, node_embed

class NodeTextCLModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        text_embedding=CFG.text_embedding,
        node_embedding=CFG.node_embedding_dim,
    ):
        super().__init__()

        self.text_encoder = TextEncoder()
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)

        self.node_encoder_prune = NodeEncoder()
        self.node_embed_prune_projection = ProjectionHead(embedding_dim=node_embedding)

        self.node_encoder = NodeEncoder()
        self.node_embed_projection = ProjectionHead(embedding_dim=node_embedding)


        self.temperature = temperature

    def forward(self, batch,feature,adj):

        # Getting Image and Text Features
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        node_embed_prune = self.node_encoder_prune(feature.float(), adj)

        # Getting Text Embeddings
        text_embeddings = self.text_projection(text_features)
        node_embeddings_project_prune = self.node_embed_prune_projection(node_embed_prune)

        node_embedding_batch_prune = node_embeddings_project_prune[batch['entity']]
        # Calculating the intra-modality Loss
        logits_prune = (node_embedding_batch_prune @ text_embeddings.T) / self.temperature
        text_similarity = text_embeddings @ text_embeddings.T
        nodes_similarity_prune = node_embedding_batch_prune @ node_embedding_batch_prune.T
        targets = F.softmax(
            (text_similarity + nodes_similarity_prune) / 2 * self.temperature, dim=-1
        )
        nodes_loss = cross_entropy(logits_prune, targets, reduction='none')
        text_loss = cross_entropy(logits_prune.T, targets.T, reduction='none')
        loss = (text_loss + nodes_loss) / 2.0  # shape: (batch_size)

        # Getting non-pruned node embeddings

        node_embed = self.node_encoder(feature.float(), adj)
        node_embeddings_project = self.node_embed_projection(node_embed)
        node_embedding_batch = node_embeddings_project[batch['entity']]

        # Getting the intra-modality Loss
        logits_intra = (node_embedding_batch @ node_embedding_batch_prune.T) / self.temperature
        node_similarity_prune = node_embedding_batch_prune @ node_embedding_batch_prune.T

        nodes_similarity = node_embedding_batch @ node_embedding_batch.T

        targets_intra = F.softmax(
            (node_similarity_prune + nodes_similarity) / 2 * self.temperature, dim=-1
        )
        nodes_loss_intra = cross_entropy(logits_intra, targets_intra, reduction='none')
        node_loss_prune_intra = cross_entropy(logits_intra.T, targets_intra.T, reduction='none')
        loss_intra = (nodes_loss_intra + node_loss_prune_intra) / 2.0  # shape: (batch_size)


        loss = loss.mean() + loss_intra.mean()

        return loss,node_embed_prune,node_embed

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
