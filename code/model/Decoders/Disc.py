import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from model.rnn import RNNEncoder, max_along_time, mean_along_time
from model.modules import CharMatching, ContextMatching

class Disc(nn.Module):
    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()
        D = n_dim
        V = len(vocab)
        self.num_choice = 5
        image_dim = args.image_dim

        self.answer_rnn = nn.LSTM(300, 300, 1, batch_first=True, dropout=0)
        #self.answer_rnn = nn.LSTM(768, 768, 1, batch_first=True, dropout=0)
        #self.answer_linear = nn.Linear(768, 300)

    def forward(self, enc_out, que, answers, **features):
        # Extract encoded context from encoder output
        #context, addit, concat_qa, q_c, e_ans, e_ans_list = enc_out
        context, concat_qa, q_c, e_ans, e_ans_list = enc_out

        # Get params
        batch_size = que.shape[0]
        q_len = features['que_len']
        ans_len = features['ans_len'].transpose(0, 1)
        #hidden_dim = context.shape[1]
        hidden_dim = context.shape[-1]
        answer_dim = e_ans.shape[3]
        num_options = self.num_choice

        # reshape to feed into answer RNN
        e_ans = e_ans.reshape(batch_size * num_options, -1, answer_dim)
        answers, _ = self.answer_rnn(e_ans)

        # keep last state and reshape back 
        answers = answers[:,-1,:]
        answers = answers.reshape(num_options, batch_size, answer_dim)
        #answers = self.answer_linear(answers)
        
        # only keep [CLS] state of BERT output
        #e_ans = e_ans[:,:,0,:]
        #answers = self.answer_linear(e_ans)

        # batch first
        answers = answers.permute(1,0,2)

        # shape the context so it is the same as the answers
        context = context.unsqueeze(dim=1).repeat(1, num_options, 1)

        answers = answers.contiguous().view(batch_size * num_options, hidden_dim)
        context = context.contiguous().view(batch_size * num_options, hidden_dim)

        # compute scores
        scores = torch.sum(answers * context, 1)
        scores = scores.view(batch_size, num_options)

        #addit = addit.transpose(0, 1)
        #addit = addit.contiguous().view(batch_size * num_options, hidden_dim)
        #scores = torch.sum(addit * context, 1)
        #scores = scores.view(batch_size, num_options)
       
        # softmax for normalized probability scores
        #scores = F.softmax(scores, dim=1)

        return scores

        def load_embedding(self, pretrained_embedding):
            print('Load pretrained embedding ...')
            self.embedding.weight.data.copy_(pretrained_embedding)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
