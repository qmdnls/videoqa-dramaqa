import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from model.rnn import RNNEncoder, max_along_time, mean_along_time
from model.modules import CharMatching, ContextMatching

class SumDisc(nn.Module):
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
        S, M, B, Q, addit, concat_qa, q_c, e_ans, e_ans_list = enc_out

        #print(S.size(), M.size(), B.size(), Q.size(), addit.size())

        # Get params
        batch_size = que.shape[0]
        q_len = features['que_len']
        ans_len = features['ans_len'].transpose(0, 1)
        #hidden_dim = context.shape[-1]
        hidden_dim = S.shape[1]
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
        S = S.unsqueeze(dim=1).repeat(1, num_options, 1)
        M = M.unsqueeze(dim=1).repeat(1, num_options, 1)
        B = B.unsqueeze(dim=1).repeat(1, num_options, 1)
        Q = Q.unsqueeze(dim=1).repeat(1, num_options, 1)
        #addit = addit.unsqueeze(dim=1).repeat(1, num_options, 1) # uncomment this for addit score

        answers = answers.contiguous().view(batch_size * num_options, hidden_dim)
        S = S.contiguous().view(batch_size * num_options, hidden_dim)
        M = M.contiguous().view(batch_size * num_options, hidden_dim)
        B = B.contiguous().view(batch_size * num_options, hidden_dim)
        Q = Q.contiguous().view(batch_size * num_options, hidden_dim)
        addit = addit.transpose(0, 1)
        addit = addit.contiguous().view(batch_size * num_options, hidden_dim)

        # compute scores
        #s_scores = torch.sum(answers * S, 1)
        s_scores = torch.sum(addit * S, 1)
        s_scores = s_scores.view(batch_size, num_options)
        s_scores = F.softmax(s_scores, dim=1)
        m_scores = torch.sum(addit * M, 1)
        m_scores = m_scores.view(batch_size, num_options)
        m_scores = F.softmax(m_scores, dim=1)
        b_scores = torch.sum(addit * B, 1)
        b_scores = b_scores.view(batch_size, num_options)
        b_scores = F.softmax(b_scores, dim=1)
        #q_scores = torch.sum(addit * Q, 1)
        #q_scores = q_scores.view(batch_size, num_options)
        #q_scores = F.softmax(q_scores, dim=1)
        #addit_scores = torch.sum(answers * addit, 1)
        #addit_scores = addit_scores.view(batch_size, num_options)
        #addit_scores = F.softmax(addit_scores, dim=1)
        #scores = s_scores + m_scores + b_scores + q_scores + addit_scores
        scores = s_scores + m_scores + b_scores

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
