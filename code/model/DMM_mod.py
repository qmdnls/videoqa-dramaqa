import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from . rnn import RNNEncoder, max_along_time
from . modules import CharMatching, ContextMatching

class UtilityBlock(nn.Module):
    """Efficient attention mechanism for many utilities block implemented for the visual dialog task (here: three utilities).
    Args:
        hidden_dim: dimension of the feature vector. Also the dimension of the final context vector provided to the decoder (required).
        feedforward_dim: dimension of the hidden feedforward layer, implementation details from "Attention is all you need" (default=2048).
        nhead: the number of heads in the multihead attention layers (default=8).
        dropout: the dropout probability (default=0.1).
    """
    def __init__(self, hidden_dim, feedforward_dim=2048, nhead=5, dropout=0.1):
        super(UtilityBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead) # dropout? separate attention modules?
        self.linear1 = nn.Linear(4*hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.relu1 = nn.ReLU(feedforward_dim)
        self.relu2 = nn.ReLU(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm([hidden_dim], elementwise_affine=False)

    def forward(self, target, source_a, source_b, source_c):
        """Passes the inputs through the utility attention block. For a detailed description see the paper. Inputs are tensors for each utility. The output is the updated utility tensor.
        Args:
            target: the target utility. The output will be of the same shape as this target utility.
            source_a: the first source utility to attend to.
            source_b: the second source utility to attend to.
        """
        # Permute to fit multihead attention input
        target = target.permute(1,0,2)
        source_a = source_a.permute(1,0,2)
        source_b = source_b.permute(1,0,2)
        source_c = source_c.permute(1,0,2)

        # Apply multihead attention mechanism for target and multiple sources as described in the paper
        out_t, _ = self.multihead_attn(target, target, target) # self attention for target utility
        out_a, _ = self.multihead_attn(target, source_a, source_a) # attention to source utility a
        out_b, _ = self.multihead_attn(target, source_b, source_b) # attention to source utility b
        out_c, _ = self.multihead_attn(target, source_c, source_c) # attention to source utility c
        
        # Permute back to batch-first
        target = target.permute(1,0,2)
        out_t = out_t.permute(1,0,2)
        out_a = out_a.permute(1,0,2)
        out_b = out_b.permute(1,0,2)
        out_c = out_c.permute(1,0,2)
        
        out = torch.cat((out_t, out_a, out_b, out_c), dim=2) # concatenate the resulting output tensors
        out = self.dropout(out)
        out = self.relu1(self.linear1(out)) 
        out = self.dropout(out)
        out = self.relu2(self.linear2(out))
        out = self.norm(out + target) # add & norm (residual target)
        return out

class UtilityLayer(nn.Module):
    """Efficient attention mechanism for many utilities layer implemented for the visual dialog task (here: three utilities). The layer consist of three parallel utility attention blocks.
    Args:
        hidden_dim: dimension of the feature vector. Also the dimension of the final context vector provided to the decoder (required).
        feedforward_dim: dimension of the hidden feedforward layer, implementation details from "Attention is all you need" (default=2048).
        nhead: the number of heads in the multihead attention layers (default=8).
        dropout: the dropout probability (default=0.1).
    """
    def __init__(self, hidden_dim, feedforward_dim=2048, nhead=5, dropout=0.1):
        super(UtilityLayer, self).__init__()
        self.utility_s = UtilityBlock(hidden_dim, feedforward_dim, nhead, dropout)
        self.utility_m = UtilityBlock(hidden_dim, feedforward_dim, nhead, dropout)
        self.utility_b = UtilityBlock(hidden_dim, feedforward_dim, nhead, dropout)
        self.utility_q = UtilityBlock(hidden_dim, feedforward_dim, nhead, dropout)

    def forward(self, S, M, B, Q):
        """Passes the input utilities through the utility attention layer. Inputs are passed through their respective blocks in parallel. The output are the three updated utility tensors.
        Args:
            V: the visual utility tensor
            Q: the question utility tensor
            R: the history utility tensor
        """
        #print(S.size(), M. size(), B.size(), Q.size())
        S_out = self.utility_s(S, M, B, Q)
        M_out = self.utility_m(M, S, B, Q)
        B_out = self.utility_b(B, S, M, Q)
        Q_out = self.utility_q(Q, S, M, B)
        return S_out, M_out, B_out, Q_out

def sum_attention(nnet, query, value, mask=None, dropout=None, mode='1D'):
	if mode == '2D':
		batch, dim = query.size(0), query.size(1)
		query = query.permute(0, 2, 3, 1).view(batch, -1, dim)
		value = value.permute(0, 2, 3, 1).view(batch, -1, dim)
		mask = mask.view(batch, 1, -1)

	scores = nnet(query).transpose(-2, -1)
	if mask is not None:
		scores.data.masked_fill_(mask.eq(0), -65504.0)

	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	weighted = torch.matmul(p_attn, value)

	return weighted, p_attn

class SummaryAttn(nn.Module):
	def __init__(self, dim, num_attn, dropout, is_multi_head=False, mode='1D'):
		super(SummaryAttn, self).__init__()
		self.linear = nn.Sequential(
			nn.Linear(dim, dim),
			nn.ReLU(inplace=True),
			nn.Linear(dim, num_attn),
		)
		self.h = num_attn
		self.is_multi_head = is_multi_head
		self.attn = None
		self.dropout = nn.Dropout(p=dropout) if dropout else None
		self.mode = mode

	def forward(self, query, value, mask=None):
		if mask is not None:
			mask = mask.unsqueeze(1)
		batch = query.size(0)

		weighted, self.attn = sum_attention(self.linear, query, value, mask=mask, dropout=self.dropout, mode=self.mode)
		weighted = weighted.view(batch, -1) if self.is_multi_head else weighted.mean(dim=-2)

		return weighted

class DMM_mod(nn.Module):
    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()
        print("Model name: Dual Matching Multistream (modified)")
        self.vocab = vocab
        V = len(vocab)
        D = n_dim

        self.embedding = nn.Embedding(V, D)
        n_dim = args.n_dim
        image_dim = args.image_dim

        self.cmat = ContextMatching(n_dim * 3) 
        self.lstm_raw = RNNEncoder(300, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        self.lstm_script = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        self.script_on = "script" in args.stream_type
        self.vbb_on = "visual_bb" in args.stream_type
        self.vmeta_on = "visual_meta" in args.stream_type
        self.conv_pool = Conv1d(n_dim*4+1, n_dim*2)

        self.util = UtilityLayer(300)

        self.character = nn.Parameter(torch.randn(22, D, device=args.device, dtype=torch.float), requires_grad=True)
        self.norm1 = Norm(D)

        self.output = nn.Linear(300, 5)

        if self.script_on:
            self.lstm_script = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_script = nn.Sequential(nn.Linear(n_dim*2, 1), nn.Softmax(dim=1))
            self.mhattn_script = CharMatching(4, D, D)

        if self.vmeta_on:            
            self.lstm_vmeta = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_vmeta = nn.Sequential(nn.Linear(n_dim*2, 1), nn.Softmax(dim=1))
            self.mhattn_vmeta = CharMatching(4, D, D)

        if self.vbb_on:
            self.lstm_vbb = RNNEncoder(image_dim+21, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
            self.vbb_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(image_dim, n_dim),
                nn.Tanh(),
            )
            self.classifier_vbb = nn.Sequential(nn.Linear(n_dim*2, 1), nn.Softmax(dim=1))

            self.mhattn_vbb = CharMatching(4, D, D)



    def _to_one_hot(self, y, n_dims, mask, dtype=torch.cuda.FloatTensor):
        scatter_dim = len(y.size())
        y_tensor = y.type(torch.LongTensor).view(*y.size(), -1).cuda()
        zeros = torch.zeros(*y.size(), n_dims).type(dtype).cuda()
        out = zeros.scatter(scatter_dim, y_tensor, 1)

        out_mask,_ = self.len_to_mask(mask, out.shape[1])
        out_mask = out_mask.unsqueeze(2).repeat(1, 1, n_dims)

        return out.masked_fill_(out_mask, 0)


    def load_embedding(self, pretrained_embedding):
        print('Load pretrained embedding ...')
        #self.embedding.weight.data.copy_(pretrained_embedding)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab, args.n_dim, args.image_dim, args.layers, args.dropout)

    def len_to_mask(self, lengths, len_max):
        #len_max = lengths.max().item()
        mask = torch.arange(len_max, device=lengths.device,
                        dtype=lengths.dtype).expand(len(lengths), len_max) >= lengths.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=torch.uint8, device=lengths.device)

        return mask, len_max

    def forward(self, que, answers, **features):
        '''
        filtered_sub (B, max_sub_len)
        filtered_sub_len (B)
        filtered_speaker (B, max_sub_len)

        filtered_visual (B, max_v_len*3)
        filtered_visual_len (B)

        filtered_image (B, max_v_len, 512)
        filtered_image_len (12)

        que (B, max_que_len)
        que_len (B)

        answers (B, 5, max_ans_len)
        ans_len (B, 5)
        
        print(que.shape)
        print(answers.shape)
        for key, value in features.items():
            print(key, value.shape)
            

        '''
        batch_size = que.shape[0]

        # -------------------------------- #
        e_q = self.embedding(que)
        q_len = features['que_len']
        e_q, _ = self.lstm_raw(e_q, q_len)
        #print("Q:", que.size(), "q_len:", q_len.size(), "e_q:", e_q.size())

        # -------------------------------- #
        e_ans = self.embedding(answers).transpose(0, 1)
        ans_len = features['ans_len'].transpose(0, 1)
        e_ans_list = [self.lstm_raw(e_a, ans_len[idx])[0] for idx, e_a in enumerate(e_ans)]


        concat_qa = [(self.get_name(que, q_len) + self.get_name(answers.transpose(0,1)[i], ans_len[i])).type(torch.cuda.FloatTensor) for i in range(5)]
        concat_qa_none = [(torch.sum(concat_qa[i], dim=1) == 0).unsqueeze(1).type(torch.cuda.FloatTensor) for i in range(5)]
        concat_qa_none = [torch.cat([concat_qa[i], concat_qa_none[i]], dim=1) for i in range(5)]
        q_c = [torch.matmul(concat_qa_none[i], self.character) for i in range(5)]
        q_c = [self.norm1(q_c[i]) for i in range(5)]
        
        if self.script_on:
            e_s = self.embedding(features['filtered_sub'])
            s_len = features['filtered_sub_len']

            # -------------------------------- #
            spk = features['filtered_speaker']
            spk_onehot = self._to_one_hot(spk, 21, mask=s_len)
            e_s = torch.cat([e_s, spk_onehot], dim=2)

            spk_flag = [torch.matmul(spk_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            spk_flag = [(spk_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            H_S, _ = self.lstm_script(e_s, s_len)
            #print("e_s:", e_s.size(), "s_len:", s_len.size(), "H_S:", H_S.size())
            #o_s = self.stream_processor(self.classifier_script,self.mhattn_script, spk_flag, H_S, s_len, q_c, e_q, q_len, e_ans_list, ans_len)
        #else:
            #o_s = 0

        if self.vmeta_on:
            vmeta = features['filtered_visual'].view(batch_size, -1, 3)
            vmeta_len = features['filtered_visual_len']*2/3

            vp = vmeta[:,:,0]
            vp = vp.unsqueeze(2).repeat(1,1,2).view(batch_size, -1)
            vbe = vmeta[:,:,1:3].contiguous()
            vbe = vbe.view(batch_size, -1)
            e_vbe = self.embedding(vbe)
            # -------------------------------- #
            vp_onehot = self._to_one_hot(vp, 21, mask=vmeta_len)
            e_vbe = torch.cat([e_vbe, vp_onehot], dim=2)
            vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            H_M, _ = self.lstm_vmeta(e_vbe, vmeta_len)
            #o_m = self.stream_processor(self.classifier_vmeta, self.mhattn_vmeta, vp_flag, H_M, vmeta_len, q_c, e_q, q_len, e_ans_list, ans_len)
        #else:
            #o_m = 0

        if self.vbb_on:
            e_vbb = features['filtered_person_full']
            vbb_len = features['filtered_person_full_len']

            vp = features['filtered_visual'].view(batch_size, -1, 3)[:,:,0]
            vp = vp.unsqueeze(2).view(batch_size, -1)
            # -------------------------------- #
            vp_onehot = self._to_one_hot(vp, 21, mask=vbb_len)
            #e_vbb =self.vbb_fc(vbb)
            e_vbb = torch.cat([e_vbb, vp_onehot], dim=2)
            vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            # -------------------------------- #
            H_B, _ = self.lstm_vbb(e_vbb, vbb_len)
            #o_b = self.stream_processor(self.classifier_vbb, self.mhattn_vbb, vp_flag, H_B, vbb_len, q_c, e_q, q_len, e_ans_list, ans_len)
        #else:
            #o_b = 0

        #out = o_s + o_m + o_b

        S = H_S
        M = H_M
        B = H_B
        Q = e_q
        S, M, B, Q = self.util(S, M, B, Q)
        #out = torch.cat((S, M, B, Q), dim=1)
        #out = self.output(out)
      
        #print("H_S:", H_S.size(), "S:", S.size())
        #print("H_M:", H_M.size(), "M:", M.size())
        #print("H_B:", H_B.size(), "B:", B.size())
        #print("e_q:", e_q.size(), "Q:", Q.size())

        # Decoder?
        o_s = self.stream_processor(self.classifier_script,self.mhattn_script, spk_flag, S, s_len, q_c, Q, q_len, e_ans_list, ans_len)
        
        vp = vmeta[:,:,0]
        vp = vp.unsqueeze(2).repeat(1,1,2).view(batch_size, -1)
        vbe = vmeta[:,:,1:3].contiguous()
        vbe = vbe.view(batch_size, -1)
        e_vbe = self.embedding(vbe)
        vp_onehot = self._to_one_hot(vp, 21, mask=vmeta_len)
        e_vbe = torch.cat([e_vbe, vp_onehot], dim=2)
        vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
        vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
        o_m = self.stream_processor(self.classifier_vmeta, self.mhattn_vmeta, vp_flag, M, vmeta_len, q_c, Q, q_len, e_ans_list, ans_len)
       
        e_vbb = features['filtered_person_full']
        vbb_len = features['filtered_person_full_len']
        vp = features['filtered_visual'].view(batch_size, -1, 3)[:,:,0]
        vp = vp.unsqueeze(2).view(batch_size, -1)
        vp_onehot = self._to_one_hot(vp, 21, mask=vbb_len)
        e_vbb = torch.cat([e_vbb, vp_onehot], dim=2)
        vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
        vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
        o_b = self.stream_processor(self.classifier_vbb, self.mhattn_vbb, vp_flag, B, vbb_len, q_c, Q, q_len, e_ans_list, ans_len)
       
        out = o_s + o_m + o_b

        return out.squeeze()

       
    def stream_processor(self, classifier, mhattn, ctx_flag, ctx, ctx_l,
                         qa_character, q_embed, q_l, a_embed, a_l):
        
        #print(ctx.size())
        #print(ctx_l.size())
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

