import torch
import torch.nn as nn
from torch.nn import functional as F

from .embedding import EmbeddingLayer
from .fc import FCLayer
from .attention import DinAttentionLayer


class DeepInterestNetwork(nn.Module):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_DIM=[162,200,80,2]):
        super(DeepInterestNetwork, self).__init__()
        self.embedding_dim = EMBEDDING_DIM
        self.hid_dim = HIDDEN_DIM

        # embeddings
        self.uid_embeddings = EmbeddingLayer(n_uid, self.embedding_dim)
        self.mid_embeddings = EmbeddingLayer(n_mid, self.embedding_dim)
        self.cat_embeddings = EmbeddingLayer(n_cat, self.embedding_dim)

        self.attn = DinAttentionLayer(embedding_dim=self.embedding_dim*2)
        mlp_input_dim = self.embedding_dim * 9
        self.mlp = nn.Sequential(
            FCLayer(mlp_input_dim, hidden_size=self.hid_dim[1], bias=True, batch_norm=True, activation='dice'),
            FCLayer(self.hid_dim[1], hidden_size=self.hid_dim[2], bias=True, activation='dice'),
            FCLayer(self.hid_dim[2], hidden_size=self.hid_dim[3], bias=False, activation='none')
        )
        uid_params = sum(p.numel() for p in self.uid_embeddings.parameters() if p.requires_grad)
        print(f"uid_embeddings trainable parameters: {uid_params}")
        mid_params = sum(p.numel() for p in self.mid_embeddings.parameters() if p.requires_grad)
        print(f"mid_embeddings trainable parameters: {mid_params}")
        cat_params = sum(p.numel() for p in self.cat_embeddings.parameters() if p.requires_grad)
        print(f"cat_embeddings trainable parameters: {cat_params}")
        att_params = sum(p.numel() for p in self.attn.parameters() if p.requires_grad)
        print(f"DinAttentionLayer trainable parameters: {att_params}")
        mlp_params = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        print(f"MLP trainable parameters: {mlp_params}")


    
    def forward(self, uids, mids, cats, mid_his, cat_his, mid_mask, noclk_mids, noclk_cats, use_negsampling=False):
        """input: uids, mids, cats, mid_his, cat_his, mid_mask, noclk_mids, noclk_cats
        """
        # item_eb, item_his_eb, mask
        uid_batch_eb = self.uid_embeddings(uids) # [B, emb_dim]
        mid_batch_eb = self.mid_embeddings(mids)
        cat_batch_eb = self.cat_embeddings(cats)
        mid_his_batch_eb = self.mid_embeddings(mid_his) # [128, 100, 18]
        cat_his_batch_eb = self.cat_embeddings(cat_his)
        
        item_eb = torch.concat((mid_batch_eb, cat_batch_eb), 1) # [128, 36]
        item_his_eb = torch.concat((mid_his_batch_eb, cat_his_batch_eb), 2) # [128, 100, 36]
        item_his_eb_sum = torch.sum(item_his_eb, dim=1) # [128, 36]
        
        if use_negsampling:
            noclk_mid_his_batch_eb = self.mid_embeddings(noclk_mids)
            noclk_cat_his_batch_eb = self.cat_embeddings(noclk_cats)
            noclk_item_his_eb = torch.concat((noclk_mid_his_batch_eb[:, :, 0, :], noclk_cat_his_batch_eb[:, :, 0, :]), -1)
            noclk_item_his_eb = noclk_item_his_eb.reshape(-1, noclk_mid_his_batch_eb.shape[1], 36)
            noclk_his_eb = torch.concat((noclk_mid_his_batch_eb, noclk_cat_his_batch_eb), -1)
            noclk_his_eb_sum_1 = torch.sum(noclk_his_eb, dim=2)
            noclk_his_eb_sum = torch.sum(noclk_his_eb_sum_1, 1)
        
        attention_output = self.attn(item_eb, item_his_eb, mid_mask) # [128, 1, 36]
        att_fea = torch.sum(attention_output, dim=1)
        inp = torch.concat((uid_batch_eb, item_eb, item_his_eb_sum, item_eb * item_his_eb_sum, att_fea), dim=-1) # [128, 162]

        y_hat = F.softmax(self.mlp(inp), dim=-1)

        return y_hat


if __name__ == "__main__":
    B = 128
    sl = 100
    uids = torch.rand((B))
    mids = torch.rand_like(uids)
    cats = torch.rand_like(uids)
    mid_his = torch.rand((B, sl))
    cat_his = torch.rand_like(mid_his)
    mid_mask = torch.ones_like(mid_his)
    noclk_mids = torch.rand((B, sl, 5))
    noclk_cats = torch.rand_like(noclk_mids)

    model = DeepInterestNetwork(n_uid=543060, n_mid=367983, n_cat=1601, EMBEDDING_DIM=12)

    y = model(uids, mids, cats, mid_his, cat_his, mid_mask, noclk_mids, noclk_cats)
    print(y.shape)
