import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, num_emb, embedding_dim):
        super(EmbeddingLayer, self).__init__()

        self.embeddings = nn.Embedding(num_emb, embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, batch_cat):
        batch_embedding = self.embeddings(batch_cat)
        return batch_embedding




if __name__ == "__main__":
    a = EmbeddingLayer(10, 12)
    b = torch.ones((2048,)).type(torch.LongTensor)
    print(a(b).size())