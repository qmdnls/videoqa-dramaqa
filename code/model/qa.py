import torch
import torch.nn as nn
from .mlp import MLP

import torch.nn.functional as F
import math
from .rnn import mean_along_time
from transformers import BertModel, BertConfig

class QA(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        V, D = vocab.shape
        D = 768
        #bert_vocab_size = 30525#30543
        self.bert_dim = 768 
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.bert.resize_token_embeddings(bert_vocab_size)

        device = args.device
        self.device = device
        self.mlp = nn.ModuleList([MLP(2 * D, 1, 50, 2) for i in range(5)])
        # self.mlp = nn.ModuleList([
        #     MLP(2 * D, 1, 50, 2),
        #     MLP(2 * D, 1, 50, 2),
        #     MLP(2 * D, 1, 50, 2),
        #     MLP(2 * D, 1, 50, 2),
        #     MLP(2 * D, 1, 50, 2)
        # ])
        self.to(device)

    def load_embedding(self, pretrained_embedding):
        print('Load pretrained embedding ...')
        # self.embedding.weight.data.copy_(pretrained_embedding)
        #self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab)

    def forward(self, que, answers, **features):
        batch_size = que.size(0)
        
        q = que
        a = answers
        print("Q:", q.size())
        print("A:", a.size())
        ql = features['que_len']
        al = features['ans_len']
    
         
        q_emb, _ = self.bert(q)
        q = q_emb[:,0,:]

        bert_output = []
        for i in range(5):
            embedded, pooled = self.bert(a[:,i,:])
            bert_output.append(embedded[:,0,:])
        a = torch.stack(bert_output)

        print("q bert out:", q.size())
        print("a bert out:", a.size())

        q = q.unsqueeze(0).repeat(5, 1, 1)
        print("pre-cat:", q.size(), a.size())
        qa = torch.cat([q, a], dim=2)

        out = torch.zeros(batch_size, 5).to(self.device)
        for i in range(5):
            out[:, i] = self.mlp[i](qa[i]).squeeze(1)
        print(out)
        return out
