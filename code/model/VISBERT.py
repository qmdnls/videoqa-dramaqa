import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel
import math
from model.rnn import RNNEncoder, max_along_time, mean_along_time
from model.modules import CharMatching, ContextMatching
from transformers import BertModel
from transformers import RobertaModel

class VISBERT(nn.Module):
    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()
        self.vocab = vocab
        V = len(vocab)
        D = n_dim

        # set appropriate CLS, SEP tokens here (from BERT tokenizer)
        self.CLS = 101
        self.SEP = 102

        self.hidden_dim = n_dim

        self.embedding = nn.Embedding(V, D)
        n_dim = args.n_dim
        image_dim = args.image_dim

        #bert_vocab_size = 30543
        self.bert_dim = 768
        #self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = RobertaModel.from_pretrained('roberta-base')
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

        self.character = nn.Parameter(torch.randn(22, D, device=args.device, dtype=torch.float), requires_grad=True)
        self.norm1 = Norm(D)

        self.visual_projection = nn.Sequential(
                nn.Linear(512, 300),
                nn.ReLU())

        self.person_projection = nn.Sequential(
                nn.Linear(512, 300),
                nn.ReLU())
        
        self.output = nn.Sequential(
                nn.Linear(768, 1),
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

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab, args.n_dim, args.image_dim, args.layers, args.dropout)


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
        ##self.embedding.weight.data.copy_(pretrained_embedding)
        #self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

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

        '''
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
     
        #concat_s = torch.stack([torch.cat([s_q, s_a[i], s_char[i], speaker_flag[i]], dim=-1) for i in range(5)], 0)
        #concat_m = torch.stack([torch.cat([m_q, m_a[i], m_char[i], meta_vp_flag[i]], dim=-1) for i in range(5)], 0)
        #concat_b = torch.stack([torch.cat([b_q, b_a[i], b_char[i], bb_vp_flag[i]], dim=-1) for i in range(5)], 0)
        
        concat_s = torch.stack([torch.cat([s_q, s_a[i], speaker_flag[i]], dim=-1) for i in range(5)], 0)
        concat_m = torch.stack([torch.cat([m_q, m_a[i], meta_vp_flag[i]], dim=-1) for i in range(5)], 0)
        concat_b = torch.stack([torch.cat([b_q, b_a[i], bb_vp_flag[i]], dim=-1) for i in range(5)], 0)
        
        concat_s = torch.mean(concat_s, dim=2)
        concat_m = torch.mean(concat_m, dim=2)
        concat_b = torch.mean(concat_b, dim=2)
        '''

        qa = features["qa"]
        sub = features["filtered_sub"]
        qa = qa.permute(1,0,2) # num_answers first
        answers = answers.permute(1,0,2) # num_answers first
        
        #cls = torch.tensor([self.CLS]).cuda().unsqueeze(0).repeat(batch_size, 1)
        #qa = []
        #for i in range(5): # for each answer
        #    concat = torch.cat([cls, sub[:,1:-1], que[:,1:], answers[i,:,1:]], dim=1)
        #    qa.append(concat)
        #qa = torch.stack(qa)

        bert_qa = []
        for i in range(5): # for each answer
            embedded, pooled = self.bert(qa[i,:,:])
            bert_qa.append(embedded[:,0,:]) # only keep [CLS] token

        #print(bert_qa[0].size())
        
        scores = []
        for i in range(5): # compute score for each answer
            out = self.output(bert_qa[i])
            scores.append(out)

        scores = torch.stack(scores, dim=1).squeeze()
        scores = F.softmax(scores, dim=1)
        #print("scores:", scores.size())
        return scores


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
