import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel
import math
from model.rnn import RNNEncoder, max_along_time, mean_along_time
from model.modules import CharMatching, ContextMatching
from transformers import BertModel


class MHA(nn.Module):
    """This module perform MultiHeadAttention for 2 utilities X, and Y as follows:
    MHA_Y(X) = MHA(X, Y, Y) and
    MHA_X(Y) = MHA(Y, X, X).
    This can be done with sharing similarity matrix since
        X_query = X_key = X_value
        Y_query = Y_key = Y_value
        Then sim_matrix(X_query, Y_key) = sim_matrix(Y_query, X_key)
    Please refer to our paper and supplementary for more details.
    """

    def __init__(self, hidden_dim, pad_size, n_head=8):
        super().__init__()

        self.hidden_size = hidden_dim
        self.num_heads = n_head
        self.pad_size = pad_size
        self.d_h = self.hidden_size // self.num_heads

        self.pad_x = torch.empty(self.pad_size, self.hidden_size)
        self.pad_x = nn.Parameter(nn.init.kaiming_uniform_(self.pad_x))
        self.pad_y = torch.empty(self.pad_size, self.hidden_size)
        self.pad_y = nn.Parameter(nn.init.kaiming_uniform_(self.pad_y))

        self.attn_X_guided_by_Y = None
        self.attn_Y_guided_by_X = None

    def project(self, X, pad_x):
        """
        Project X into X_query, X_key, X_value (all are X_proj) by
        splitting along last indexes mechanically.
        Note that: X_query = X_key = X_value = X_proj since W_Q = W_K = W_V.
        Arguments
        ---------
        X: torch.FloatTensor
            The input tensor with
            Shape [batch_size, M, hidden_size]
        pad_x: torch.FloatTensor
            The padding vectors we would like to put at the beginning of X
            Shape [batch_size, pad_size, hidden_size]
        Returns
        -------
        X_proj: torch.FloatTensor
            The summarized vector of the utility (the context vector for this utility)
            Shape [batch_size, M + pad_size, num_heads, d_h]
        """
        size = X.size(0), self.pad_size, self.hidden_size
        X = torch.cat([pad_x.unsqueeze(0).expand(*size), X], dim=1)

        X_proj = X.view(X.size(0), X.size(1), self.num_heads, self.d_h)
        return X_proj

    def forward(self, X, Y):
        """
        Arguments
        ---------
        X: torch.FloatTensor
            The input tensor of utility X
            Shape [batch_size, M, hidden_size]
        Y: torch.FloatTensor
            The input tensor of utility Y
            Shape [batch_size, N, hidden_size]
        mask_X: torch.LongTensor
            The mask of utility X where 0 denotes <PAD>
            Shape [batch_size, M]
        mask_Y: torch.LongTensor
            The mask of utility Y where 0 denotes <PAD>
            Shape [batch_size, N]
        Returns
        -------
        A tuple of two MultiHeadAttention
            A_X(Y): torch.FloatTensor
                The attention from the source Y to X: Y_attends_in_X
                Shape [batch_size, M, hidden_size]
            A_Y(X): torch.FloatTensor
                The attention from the source X to Y: X_attends_in_Y
                Shape [batch_size, N, hidden_size]
        """

        pad_mask = X.new_ones((X.size(0), self.pad_size)).long()

        mask_X = torch.ones(X.size(0), X.size(1), dtype=torch.long).cuda()
        mask_Y = torch.ones(Y.size(0), Y.size(1), dtype=torch.long).cuda()

        mask_X = torch.cat([pad_mask, mask_X], dim=1)
        mask_Y = torch.cat([pad_mask, mask_Y], dim=1)
        M_pad, N_pad = mask_X.size(1), mask_Y.size(1)
        mask_X = mask_X[:, None, :, None].repeat(1, self.num_heads, 1, N_pad)
        mask_Y = mask_Y[:, None, None, :].repeat(1, self.num_heads, M_pad, 1)

        # X_proj: [bs, pad_size + M, num_heads, d_h]
        X_proj = self.project(X, self.pad_x)

        # Y_proj [bs, pad_size + N, num_heads, d_h]
        Y_proj = self.project(Y, self.pad_y)

        # (1) shape [bs, num_heads, pad_size + M, d_h]
        # (2) shape [bs, num_heads, d_h, pad_size + N]
        X_proj = X_proj.permute(0, 2, 1, 3)
        Y_proj = Y_proj.permute(0, 2, 3, 1)

        """
        Note that:
        X_query = X_key = X_value = X_proj,
        Y_query = Y_key = Y_value = Y_proj
        Then, we have sim_matrix(X_query, Y_key) = sim_matrix(Y_query, X_key) = sim_matrix
        """
        # shape: [bs, num_heads, pad_size + M, pad_size + N]
        sim_matrix = torch.matmul(X_proj, Y_proj)
        sim_matrix = sim_matrix.masked_fill(mask_X == 0, -1e9)
        sim_matrix = sim_matrix.masked_fill(mask_Y == 0, -1e9)

        # shape: [bs, num_heads, pad_size + M, pad_size + N]
        attn_X_guided_by_Y = torch.softmax(sim_matrix, dim=2)
        attn_Y_guided_by_X = torch.softmax(sim_matrix, dim=3)

        # shape [bs, num_heads, pad_size + M, d_h]
        X_value = X_proj
        # shape [bs, num_heads, pad_size + N, d_h]
        X_attends_in_Y = torch.matmul(attn_X_guided_by_Y.transpose(2, 3), X_value)
        # shape [bs, num_heads, N, d_h]
        X_attends_in_Y = X_attends_in_Y[:, :, self.pad_size:, :]
        # shape [bs, N, num_heads, d_h]
        X_attends_in_Y = X_attends_in_Y.permute(0, 2, 1, 3).contiguous()
        # shape [bs, N, num_heads, hidden_size]
        X_attends_in_Y = X_attends_in_Y.view(X_attends_in_Y.size(0), X_attends_in_Y.size(1), -1)

        # shape [bs, num_heads, pad_size + N, d_h]
        Y_value = Y_proj.permute(0, 1, 3, 2).contiguous()
        # shape [bs, num_heads, pad_size + M, d_h]
        Y_attends_in_X = torch.matmul(attn_Y_guided_by_X, Y_value)
        # shape [bs, num_heads, M, d_h]
        Y_attends_in_X = Y_attends_in_X[:, :, self.pad_size:, :]
        # shape [bs, M, num_heads, d_h]
        Y_attends_in_X = Y_attends_in_X.permute(0, 2, 1, 3).contiguous()
        # shape [bs, M, hidden_size]
        Y_attends_in_X = Y_attends_in_X.view(Y_attends_in_X.size(0), Y_attends_in_X.size(1), -1)

        return X_attends_in_Y, Y_attends_in_X

class UtilityBlock(nn.Module):
    """Efficient attention mechanism for many utilities block implemented for the visual dialog task (here: three utilities).
    Args:
        hidden_dim: dimension of the feature vector. Also the dimension of the final context vector provided to the decoder (required).
        feedforward_dim: dimension of the hidden feedforward layer, implementation details from "Attention is all you need" (default=2048).
        n_head: the number of heads in the multihead attention layers (default=8).
        dropout: the dropout probability (default=0.1).
    """
    def __init__(self, hidden_dim, feedforward_dim=2048, n_head=8, dropout=0.1):
        super(UtilityBlock, self).__init__()
        #self.multihead_attn = nn.MultiheadAttention(hidden_dim, n_head) # dropout? separate attention modules?
        self.multihead_attn = MHA(hidden_dim, pad_size=64, n_head=n_head)
        self.linear1 = nn.Linear(4*hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        #self.relu1 = nn.ReLU(feedforward_dim)
        #self.relu2 = nn.ReLU(hidden_dim)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
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
        #target = target.permute(1,0,2)
        #source_a = source_a.permute(1,0,2)
        #source_b = source_b.permute(1,0,2)
        #source_c = source_c.permute(1,0,2)

        # Apply multihead attention mechanism for target and multiple sources as described in the paper
        #out_t, _ = self.multihead_attn(target, target, target) # self attention for target utility
        #out_a, _ = self.multihead_attn(target, source_a, source_a) # attention to source utility a
        #out_b, _ = self.multihead_attn(target, source_b, source_b) # attention to source utility b
        #out_c, _ = self.multihead_attn(target, source_c, source_c) # attention to source utility c
        
        _, out_t = self.multihead_attn(target, target) # self attention for target utility
        _, out_a = self.multihead_attn(target, source_a) # attention to source utility a
        _, out_b = self.multihead_attn(target, source_b) # attention to source utility b
        _, out_c = self.multihead_attn(target, source_c) # attention to source utility c
        
        # Permute back to batch-first
        #target = target.permute(1,0,2)
        #out_t = out_t.permute(1,0,2)
        #out_a = out_a.permute(1,0,2)
        #out_b = out_b.permute(1,0,2)
        #out_c = out_c.permute(1,0,2)
        
        out = torch.cat((out_t, out_a, out_b, out_c), dim=2) # concatenate the resulting output tensors
        out = self.relu1(self.linear1(out)) 
        out = self.dropout(out)
        out = self.relu2(self.linear2(out))
        out = self.dropout(out)
        out = self.norm(out + target) # add & norm (residual target)
        return out

class UtilityLayer(nn.Module):
    """Efficient attention mechanism for many utilities layer implemented for the visual dialog task (here: three utilities). The layer consist of three parallel utility attention blocks.
    Args:
        hidden_dim: dimension of the feature vector. Also the dimension of the final context vector provided to the decoder (required).
        feedforward_dim: dimension of the hidden feedforward layer, implementation details from "Attention is all you need" (default=2048).
        n_head: the number of heads in the multihead attention layers (default=8).
        dropout: the dropout probability (default=0.1).
    """
    def __init__(self, hidden_dim, feedforward_dim=1024, n_head=5, dropout=0.1):
        super(UtilityLayer, self).__init__()
        self.utility_s = UtilityBlock(hidden_dim, feedforward_dim, n_head, dropout)
        self.utility_m = UtilityBlock(hidden_dim, feedforward_dim, n_head, dropout)
        self.utility_b = UtilityBlock(hidden_dim, feedforward_dim, n_head, dropout)
        self.utility_q = UtilityBlock(hidden_dim, feedforward_dim, n_head, dropout)
        #self.utility_v = UtilityBlock(hidden_dim, feedforward_dim, n_head, dropout)

    def forward(self, S, M, B, Q):
        """Passes the input utilities through the utility attention layer. Inputs are passed through their respective blocks in parallel. The output are the three updated utility tensors.
        Args:
            V: the visual utility tensor
            Q: the question utility tensor
            R: the history utility tensor
        """
        S_out = self.utility_s(S, M, B, Q)
        M_out = self.utility_m(M, S, B, Q)
        B_out = self.utility_b(B, S, M, Q)
        Q_out = self.utility_q(Q, S, M, B)
        #V_out = self.utility_v(V
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
			nn.PReLU(),
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

class BERT(nn.Module):
    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()
        self.vocab = vocab
        V = len(vocab)
        D = n_dim

        self.hidden_dim = n_dim

        self.embedding = nn.Embedding(V, D)
        n_dim = args.n_dim
        image_dim = args.image_dim

        #bert_vocab_size = 30543
        self.bert_dim = 768
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.bert.resize_token_embeddings(bert_vocab_size)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.cmat = ContextMatching(n_dim * 3) 
        self.lstm_raw = RNNEncoder(self.bert_dim, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        #self.lstm_script = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        self.script_on = "script" in args.stream_type
        self.vbb_on = "visual_bb" in args.stream_type
        self.vmeta_on = "visual_meta" in args.stream_type
        self.conv_pool = Conv1d(n_dim*4+1, n_dim*2)

        self.util = UtilityLayer(hidden_dim=300, feedforward_dim=1024, n_head=6, dropout=0.5)
        self.util2 = UtilityLayer(hidden_dim=300, feedforward_dim=1024, n_head=6, dropout=0.5)
        self.summary_s = SummaryAttn(300, 8, 0.1)
        self.summary_m = SummaryAttn(300, 8, 0.1)
        self.summary_b = SummaryAttn(300, 8, 0.1)
        self.summary_q = SummaryAttn(300, 8, 0.1)
        self.summary_addit = SummaryAttn(300, 8, 0.1)

        self.character = nn.Parameter(torch.randn(22, D, device=args.device, dtype=torch.float), requires_grad=True)
        self.norm1 = Norm(D)

        self.visual_projection = nn.Sequential(
                nn.Linear(512, 300),
                nn.ReLU())

        self.person_projection = nn.Sequential(
                nn.Linear(512, 300),
                nn.ReLU())
        
        self.output = nn.Sequential(
                nn.Linear(5*300, 300),
                nn.PReLU())

        self.linear_addit = nn.Sequential(
                nn.Linear(1803, 300),
                nn.PReLU())

        if self.script_on:
            #self.lstm_script = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
            self.lstm_script = RNNEncoder(self.bert_dim + 21, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_script = nn.Sequential(nn.Linear(n_dim*2, 1), nn.Softmax(dim=1))
            self.mhattn_script = CharMatching(4, D, D)

        if self.vmeta_on:            
            #self.lstm_vmeta = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
            self.lstm_vmeta = RNNEncoder(self.bert_dim + 21, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
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

    def len_to_mask(self, lengths, len_max):
        #len_max = lengths.max().item()
        mask = torch.arange(len_max, device=lengths.device,
                        dtype=lengths.dtype).expand(len(lengths), len_max) >= lengths.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=torch.uint8, device=lengths.device)

        return mask, len_max

    def get_name(self, x, x_l):
        x_mask = x.masked_fill(x>20, 21)
        x_onehot = self._to_one_hot(x_mask, 22, x_l)
        x_sum = torch.sum(x_onehot[:,:,:21], dim=1)
        return x_sum > 0

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
        num_options = answers.shape[1]

        # -------------------------------- #
        e_q, _ = self.bert(que)
        q_len = features['que_len']
        e_q, _ = self.lstm_raw(e_q, q_len)

        # -------------------------------- #
        #ans = answers.reshape(batch_size * num_options, -1) # reshape for transformer input
        #e_ans, _ = self.bert(ans)
        #e_ans = e_ans.reshape(batch_size, num_options, -1, self.bert_dim).transpose(0, 1) # reshape back
        bertout = []
        for i in range(5):
            embedded, pooled = self.bert(answers[:,i,:])
            bertout.append(embedded)
        e_ans = torch.stack(bertout)
        
        ans_len = features['ans_len'].transpose(0, 1)
        e_ans_list = [self.lstm_raw(e_a, ans_len[idx])[0] for idx, e_a in enumerate(e_ans)]

        concat_qa = [(self.get_name(que, q_len) + self.get_name(answers.transpose(0,1)[i], ans_len[i])).type(torch.cuda.FloatTensor) for i in range(5)]
        concat_qa_none = [(torch.sum(concat_qa[i], dim=1) == 0).unsqueeze(1).type(torch.cuda.FloatTensor) for i in range(5)]
        concat_qa_none = [torch.cat([concat_qa[i], concat_qa_none[i]], dim=1) for i in range(5)]
        q_c = [torch.matmul(concat_qa_none[i], self.character) for i in range(5)]
        q_c = [self.norm1(q_c[i]) for i in range(5)]


        # Get image features
        image = features['filtered_image']
        person_features = features['filtered_person_full']
        #print("Image:", image.size())
        #print("Person features:", person_features.size())
        #print(stop_code_here)

        
        if self.script_on:
            #e_s = self.embedding(features['filtered_sub'])
            e_s, _ = self.bert(features['filtered_sub'])
            s_len = features['filtered_sub_len']

            # -------------------------------- #
            spk = features['filtered_speaker']
            spk_onehot = self._to_one_hot(spk, 21, mask=s_len)
            e_s = torch.cat([e_s, spk_onehot], dim=2)

            spk_flag = [torch.matmul(spk_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            spk_flag = [(spk_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            H_S, _ = self.lstm_script(e_s, s_len)

        if self.vmeta_on:
            vmeta = features['filtered_visual'].view(batch_size, -1, 3)
            vmeta_len = features['filtered_visual_len']*2/3

            vp = vmeta[:,:,0]
            vp = vp.unsqueeze(2).repeat(1,1,2).view(batch_size, -1)
            vbe = vmeta[:,:,1:3].contiguous()
            vbe = vbe.view(batch_size, -1)
            #e_vbe = self.embedding(vbe)
            e_vbe, _ = self.bert(vbe)
            # -------------------------------- #
            vp_onehot = self._to_one_hot(vp, 21, mask=vmeta_len)
            e_vbe = torch.cat([e_vbe, vp_onehot], dim=2)
            vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            H_M, _ = self.lstm_vmeta(e_vbe, vmeta_len)

        if self.vbb_on:
            e_vbb = features['filtered_person_full']
            vbb_len = features['filtered_person_full_len']

            vp = features['filtered_visual'].view(batch_size, -1, 3)[:,:,0]
            vp = vp.unsqueeze(2).view(batch_size, -1)
            vp_onehot = self._to_one_hot(vp, 21, mask=vbb_len)
            e_vbb = torch.cat([e_vbb, vp_onehot], dim=2)
            vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]


            vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            H_B, _ = self.lstm_vbb(e_vbb, vbb_len)


        S = H_S
        M = H_M
        B = H_B
        #Q = e_q
        Q = torch.stack([q_c[i] for i in range(5)], dim=1)

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

        s_q, s_a, s_char = self.processor(S, sub_len, q_c, e_q, q_len, e_ans_list, ans_len, self.mhattn_script)
        m_q, m_a, m_char = self.processor(M, meta_len, q_c, e_q, q_len, e_ans_list, ans_len, self.mhattn_vmeta)
        b_q, b_a, b_char = self.processor(B, bb_len, q_c, e_q, q_len, e_ans_list, ans_len, self.mhattn_vbb)

        # use full image features
        B = self.person_projection(person_features)
        V = self.visual_projection(image)

        S, M, B, Q = self.util(S, M, B, Q)
        #S, M, B, Q = self.util2(S, M, B, Q)
        
        # self attention
        S = self.summary_s(S,S)
        M = self.summary_m(M,M)
        B = self.summary_b(B,B)
        Q = self.summary_q(Q,Q)
        
        #S = mean_along_time(S, s_len)
        #M = mean_along_time(M, vmeta_len)
        #B = mean_along_time(B, vbb_len)
        #Q = mean_along_time(Q, q_len) 
      
        #concat_s = torch.stack([torch.cat([s_q, s_a[i], s_char[i], speaker_flag[i]], dim=-1) for i in range(5)], 0)
        #concat_m = torch.stack([torch.cat([m_q, m_a[i], m_char[i], meta_vp_flag[i]], dim=-1) for i in range(5)], 0)
        #concat_b = torch.stack([torch.cat([b_q, b_a[i], b_char[i], bb_vp_flag[i]], dim=-1) for i in range(5)], 0)
        
        concat_s = torch.stack([torch.cat([s_q, s_a[i], speaker_flag[i]], dim=-1) for i in range(5)], 0)
        concat_m = torch.stack([torch.cat([m_q, m_a[i], meta_vp_flag[i]], dim=-1) for i in range(5)], 0)
        concat_b = torch.stack([torch.cat([b_q, b_a[i], bb_vp_flag[i]], dim=-1) for i in range(5)], 0)
        
        concat_s = torch.mean(concat_s, dim=2)
        concat_m = torch.mean(concat_m, dim=2)
        concat_b = torch.mean(concat_b, dim=2)

        addit = torch.cat([concat_s, concat_m, concat_b], dim=-1) 
        addit = self.linear_addit(addit) 
        #print(addit.size())
        addit = addit.transpose(0, 1) # batch first
        #print(addit.size())
        addit = self.summary_addit(addit,addit)
        #print(addit.size())
        # compute context vector for decoder
        c = self.output(torch.cat([S, M, B, Q, addit], dim=1))

        return c, addit, concat_qa, q_c, e_ans, e_ans_list


    def processor(self, context, context_l, qa_character, q_embed, q_l, a_embed, a_l, mhattn):
        #print(context.size(), context_l, len(qa_character), q_embed.size(), q_l, len(a_embed), a_l)
        u_q = self.cmat(context, context_l, q_embed, q_l)
        u_a = torch.stack([self.cmat(context, context_l, a_embed[i], a_l[i]) for i in range(5)])
        u_ch = torch.stack([mhattn(qa_character[i], context, context_l) for i in range(5)])
        return u_q, u_a, u_ch

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
        return out

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
