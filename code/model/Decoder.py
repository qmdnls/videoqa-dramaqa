import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from . rnn import RNNEncoder, max_along_time
from . modules import CharMatching, ContextMatching

class Decoder(nn.Module):
    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()
        self.vocab = vocab
        V = len(vocab)
        D = n_dim
        
        n_dim = args.n_dim
        image_dim = args.image_dim

        self.cmat = ContextMatching(n_dim * 3) 
        self.conv_pool = Conv1d(n_dim*4+1, n_dim*2)
        self.character = nn.Parameter(torch.randn(22, D, device=args.device, dtype=torch.float), requires_grad=True)
        self.norm1 = Norm(D)

        self.classifier_script = nn.Sequential(nn.Linear(n_dim*2, 1), nn.Softmax(dim=1))
        self.mhattn_script = CharMatching(3, D, D, dropout=0.5)

        self.classifier_vmeta = nn.Sequential(nn.Linear(n_dim*2, 1), nn.Softmax(dim=1))
        self.mhattn_vmeta = CharMatching(3, D, D, dropout=0.5)

        self.vbb_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(image_dim, n_dim),
            nn.Tanh(),
        )
        self.classifier_vbb = nn.Sequential(nn.Linear(n_dim*2, 1), nn.Softmax(dim=1))
        self.mhattn_vbb = CharMatching(3, D, D, dropout=0.5)

    def _to_one_hot(self, y, n_dims, mask, dtype=torch.cuda.FloatTensor):
        scatter_dim = len(y.size())
        y_tensor = y.type(torch.LongTensor).view(*y.size(), -1).cuda()
        zeros = torch.zeros(*y.size(), n_dims).type(dtype).cuda()
        out = zeros.scatter(scatter_dim, y_tensor, 1)
        out_mask,_ = self.len_to_mask(mask, out.shape[1])
        out_mask = out_mask.unsqueeze(2).repeat(1, 1, n_dims)
        return out.masked_fill_(out_mask, 0)

    def len_to_mask(self, lengths, len_max):
        mask = torch.arange(len_max, device=lengths.device,
                        dtype=lengths.dtype).expand(len(lengths), len_max) >= lengths.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=torch.uint8, device=lengths.device)
        return mask, len_max

    def forward(self, enc_out, que, answers, **features):
        # Extract encoded context from encoder output
        S, M, B, Q, concat_qa, q_c, e_ans, e_ans_list = enc_out
        
        # Get params
        batch_size = que.shape[0]
        q_len = features['que_len']
        ans_len = features['ans_len'].transpose(0, 1)
        
        #concat_qa = [(self.get_name(que, q_len) + self.get_name(answers.transpose(0,1)[i], ans_len[i])).type(torch.cuda.FloatTensor) for i in range(5)]
        #concat_qa_none = [(torch.sum(concat_qa[i], dim=1) == 0).unsqueeze(1).type(torch.cuda.FloatTensor) for i in range(5)]
        #concat_qa_none = [torch.cat([concat_qa[i], concat_qa_none[i]], dim=1) for i in range(5)]
        #q_c = [torch.matmul(concat_qa_none[i], self.character) for i in range(5)]
        #q_c = [self.norm1(q_c[i]) for i in range(5)]
        #e_ans = self.embedding(answers).transpose(0, 1)
        #ans_len = features['ans_len'].transpose(0, 1)
        #e_ans_list = [self.lstm_raw(e_a, ans_len[idx])[0] for idx, e_a in enumerate(e_ans)]

        # Compute speaker flag
        sub_len = features['filtered_sub_len']
        speaker = features['filtered_speaker']
        speaker_onehot = self._to_one_hot(speaker, 21, mask=sub_len)
        speaker_flag = [torch.matmul(speaker_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
        speaker_flag = [(speaker_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]

        # Compute meta flag
        meta = features['filtered_visual'].view(batch_size, -1, 3)
        meta_len = features['filtered_visual_len']*2/3
        meta_vp = meta[:,:,0]
        meta_vp = meta_vp.unsqueeze(2).repeat(1,1,2).view(batch_size, -1)
        meta_vp_onehot = self._to_one_hot(meta_vp, 21, mask=meta_len)
        meta_vp_flag = [torch.matmul(meta_vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
        meta_vp_flag = [(meta_vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]

        # Compute bounding box flag
        bb_len = features['filtered_person_full_len']
        bb_vp = features['filtered_visual'].view(batch_size, -1, 3)[:,:,0]
        bb_vp = bb_vp.unsqueeze(2).view(batch_size, -1)
        bb_vp_onehot = self._to_one_hot(bb_vp, 21, mask=bb_len)
        bb_vp_flag = [torch.matmul(bb_vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
        bb_vp_flag = [(bb_vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]

        o_s = self.stream_processor(self.classifier_script,self.mhattn_script, speaker_flag, S, sub_len, q_c, Q, q_len, e_ans_list, ans_len)
        o_m = self.stream_processor(self.classifier_vmeta, self.mhattn_vmeta, meta_vp_flag, M, meta_len, q_c, Q, q_len, e_ans_list, ans_len)
        o_b = self.stream_processor(self.classifier_vbb, self.mhattn_vbb, bb_vp_flag, B, bb_len, q_c, Q, q_len, e_ans_list, ans_len)
       
        out = o_s + o_m + o_b
        out = out.squeeze()
        return out

       
    def stream_processor(self, classifier, mhattn, ctx_flag, ctx, ctx_l,
                         qa_character, q_embed, q_l, a_embed, a_l):
        
        u_q = self.cmat(ctx, ctx_l, q_embed, q_l)
        u_a = [self.cmat(ctx, ctx_l, a_embed[i], a_l[i]) for i in range(5)]
        u_ch = [mhattn(qa_character[i], ctx, ctx_l) for i in range(5)]

        concat_a = [torch.cat([ctx,  u_q,u_a[i], u_ch[i], ctx_flag[i]], dim=-1) for i in range(5)] 
        
        # ctx, u_ch[i], ctx_flag[i],
        # exp_2 : ctx, u_a[i], u_q, ctx_flag[i], u_ch[i]
        maxout = [self.conv_pool(concat_a[i], ctx_l) for i in range(5)]

        answers = torch.stack(maxout, dim=1)
        out = classifier(answers)  # (B, 5)

        return out 

    def get_name(self, x, x_l):
        x_mask = x.masked_fill(x>20, 21)
        x_onehot = self._to_one_hot(x_mask, 22, x_l)
        x_sum = torch.sum(x_onehot[:,:,:21], dim=1)
        return x_sum > 0




class Conv1d(nn.Module):
    def __init__(self, n_dim, out_dim):
        super().__init__()
        out_dim = int(out_dim/4)
        self.conv_k1 = nn.Conv1d(n_dim, out_dim, kernel_size=1, stride=1)
        self.conv_k2 = nn.Conv1d(n_dim, out_dim, kernel_size=2, stride=1)
        self.conv_k3 = nn.Conv1d(n_dim, out_dim, kernel_size=3, stride=1)
        self.conv_k4 = nn.Conv1d(n_dim, out_dim, kernel_size=4, stride=1)
        #self.maxpool = nn.MaxPool1d(kernel_size = )

    def forward(self, x, x_l):
        # x : (B, T, 5*D)
        x_pad = torch.zeros(x.shape[0],3,x.shape[2]).type(torch.cuda.FloatTensor)
        x = torch.cat([x, x_pad], dim=1)
        x1 = F.relu(self.conv_k1(x.transpose(1,2)))[:,:,:-3]
        x2 = F.relu(self.conv_k2(x.transpose(1,2)))[:,:,:-2]
        x3 = F.relu(self.conv_k3(x.transpose(1,2)))[:,:,:-1]
        x4 = F.relu(self.conv_k4(x.transpose(1,2)))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = out.transpose(1,2)
        return max_along_time(out, x_l)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
          # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
