import torch
import math
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable
from transformers import RobertaTokenizer, RobertaModel
from model.rnn import RNNEncoder, max_along_time, mean_along_time
from model.modules import CharMatching, ContextMatching

class MMT_lm(nn.Module):
    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()
        self.vocab = vocab
        V = len(vocab)
        D = n_dim
        self.hidden_dim = n_dim
        self.num_choice = num_choice

        #video_encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=6, dim_feedforward=1024, dropout=0.1, activation='gelu')
        #self.video_encoder = nn.TransformerEncoder(video_encoder_layer, num_layers=1)
        #self.video_encoder = nn.GRU(2048, 150, bidirectional=True, batch_first=True)

        #multimodal_encoder_layer = nn.TransformerEncoderLayer(d_model=n_dim, nhead=6, dim_feedforward=1024, dropout=0.5, activation='gelu')
        #self.transformer = nn.TransformerEncoder(multimodal_encoder_layer, num_layers=2)
        #self.transformer = nn.Transformer(d_model=n_dim, nhead=6)

        self.embedding = nn.Embedding(V, D)
        n_dim = args.n_dim
        image_dim = args.image_dim

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.language_model = RobertaModel.from_pretrained('roberta-base', return_dict=True) 
        #for param in self.language_model.base_model.parameters():
        #    param.requires_grad = False

        # Update config to finetune token type embeddings
        self.language_model.config.type_vocab_size = 5 

        # Create a new Embeddings layer, with 2 possible segments IDs instead of 1
        self.language_model.embeddings.token_type_embeddings = nn.Embedding(5, self.language_model.config.hidden_size)
                
        # Initialize it
        self.language_model.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.language_model.config.initializer_range)

        # Freeze the first 6 layers
        #modules = [self.language_model.encoder.layer[:6]]
        #for module in modules:
        #    for param in module.parameters():
        #        param.requires_grad = False

        #self.cmat = ContextMatching(n_dim * 3) 
        #self.lstm_raw = RNNEncoder(300, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        #self.lstm_script = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        self.script_on = "script" in args.stream_type
        self.vbb_on = "visual_bb" in args.stream_type
        self.vmeta_on = "visual_meta" in args.stream_type
        #self.conv_pool = Conv1d(n_dim*4+1, n_dim*2)

        #self.character = nn.Parameter(torch.randn(22, D, device=args.device, dtype=torch.float), requires_grad=True)
        #self.norm1 = Norm(D)

        self.lang_proj = nn.Linear(768, 300)
        self.visual_proj = nn.Linear(2048, 300) 
        
        #self.mh_video = nn.MultiheadAttention(300, 6) 
        #self.context_gru = nn.GRU(300, 150, bidirectional=True, batch_first=True)
        self.cross1 = UtilityLayer(300)
        #self.cross2 = UtilityLayer(300)

        self.char_classifier = nn.Linear(300, 21)
        self.mask_classifier = nn.Linear(300, self.tokenizer.vocab_size)

        self.video_cls = nn.Linear(300, 1)
        self.language_cls = nn.Linear(300,1)
        #self.joint_cls = nn.Linear(300, 1)

        #self.answer_rnn = nn.LSTM(300, 300, 1, batch_first=True, dropout=0)

        speaker_name = [ 
            'None', # index 0: unknown speaker 
            'Anna', 'Chairman', 'Deogi', 'Dokyung', 'Gitae',
            'Haeyoung1', 'Haeyoung2', 'Heeran', 'Hun', 'Jeongsuk',
            'Jinsang', 'Jiya', 'Kyungsu', 'Sangseok', 'Seohee', 
            'Soontack', 'Sukyung', 'Sungjin', 'Taejin', 'Yijoon'
            ]
        self.speaker_to_index = {name: index for index, name in enumerate(speaker_name)} 
        self.index_to_speaker = {v: k for k, v in self.speaker_to_index.items()}

        '''
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
        '''


    def _to_one_hot(self, y, n_dims, mask, dtype=torch.cuda.FloatTensor):
        scatter_dim = len(y.size())
        y_tensor = y.type(torch.LongTensor).view(*y.size(), -1).cuda()
        y_tensor = y.view(*y.size(), -1).cuda()
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
        batch_size = que.shape[0]

        """
        if self.script_on:
            s_len = features['filtered_sub_len']
            spk = features['filtered_speaker']
            spk_onehot = self._to_one_hot(spk, 21, mask=s_len)
            e_s = torch.cat([e_script, spk_onehot], dim=2)
            H_S, _ = self.lstm_script(e_s, s_len)

        if self.vmeta_on:
            vmeta = features['filtered_visual'].view(batch_size, -1, 3)
            vmeta_len = features['filtered_visual_len'].double()*2/3

            vp = vmeta[:,:,0]
            vp = vp.unsqueeze(2).repeat(1,1,2).view(batch_size, -1)
            vbe = vmeta[:,:,1:3].contiguous()
            vbe = vbe.view(batch_size, -1)
            #e_vbe = self.embedding(vbe)
            e_vbe = self.language_model(vbe).last_hidden_state
            e_vbe = self.lang_proj(e_vbe)
            # -------------------------------- #
            vp_onehot = self._to_one_hot(vp, 21, mask=vmeta_len)
            e_vbe = torch.cat([e_vbe, vp_onehot], dim=2)
            #vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            #vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            H_M, _ = self.lstm_vmeta(e_vbe, vmeta_len)

        if self.vbb_on:
            e_vbb = features['filtered_person_full']
            vbb_len = features['filtered_person_full_len']

            vp = features['filtered_visual'].view(batch_size, -1, 3)[:,:,0]
            vp = vp.unsqueeze(2).view(batch_size, -1)
            vp_onehot = self._to_one_hot(vp, 21, mask=vbb_len)
            e_vbb = torch.cat([e_vbb, vp_onehot], dim=2)
            #vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            #vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            H_B, _ = self.lstm_vbb(e_vbb, vbb_len)


        S = H_S
        M = H_M
        B = H_B
        Q = e_q
        #Q = torch.stack([q_c[i] for i in range(5)], dim=1)
        #F = features['images'].squeeze()
        #video = features['filtered_image']
        #per_person_features = self.visual_proj(features['per_person_features'])
        #video = self.visual_proj(video)
        """
        
        text = features['text_masked']
        text_length = text.size(1)
        token_type_ids = features['token_type_ids']
        answer_length = answers.size(2)
        answer_type_ids = batch_size * [answer_length * [3]]
        answer_type_ids = torch.tensor(answer_type_ids, dtype=torch.long).cuda()
        token_type_ids = torch.cat([token_type_ids, answer_type_ids], dim=1)

        # encode the text using roberta
        e_ans = []
        for i in range(5):
            answer = answers[:,i,:]
            text_answer = torch.cat([text, answer], dim=-1)
            outputs = self.language_model(text_answer, token_type_ids=token_type_ids)
            embedded = outputs.last_hidden_state
            embedded = self.lang_proj(embedded)
            e_ans.append(embedded)

        # stack the text-answers pairs
        text = torch.stack(e_ans, dim=1)

        # predict the masked tokens @@ UNCOMMENT THIS
        #labels = self.mask_classifier(text[:,0,:text_length,:])

        # encode video frames
        video = features['filtered_person_full']
        video_length = video.size(1)
        video = self.visual_proj(video)
        
        # predict person contained in each bounding box @@ UNCOMMENT THIS
        #char = self.char_classifier(video)
        #char = self.char_classifier(context.unsqueeze(dim=1).repeat(1, video_length, 1))
        
        # Attend video frames to text
        #video = video.permute(1,0,2)
        #text = text.permute(1,0,2)
        #video, _ = self.mh_video(text, video, video)
        #video = video.permute(1,0,2)
        #text = text.permute(1,0,2)

        # Transformer video encoder
        #video = video.permute(1,0,2)
        #video = self.video_encoder(video)
        #video = video.permute(1,0,2)

        ### CROSS-MODALITY UTILITY LAYER
        context = []
        for i in range(5):
            text_input = text[:,i,:,:]
            text_context, video_context = self.cross1(text_input, video)
            #text_context, video_context = self.cross2(text_context, video_context)
            context.append((text_context, video_context))
            del text_input
        text_context, video_context = zip(*context)
        del context
        
        # stack text and video context and maxpool
        video_context = torch.stack(video_context, dim=1)
        video_context, _ = torch.max(video_context, dim=2)
        text_context = torch.stack(text_context, dim=1)
        label_prediction_context, _ = torch.max(text_context, dim=1)
        char_prediction_context, _ = torch.max(video_context, dim=1)
        text_context, _ = torch.max(text_context, dim=2)
     
        # predict masked tokens
        labels = self.mask_classifier(label_prediction_context[:,:text_length,:])

        # classify answers @@ UNCOMMENT THIS
        visual_score = self.video_cls(video_context).squeeze()
        language_score = self.language_cls(text_context).squeeze()
        scores = visual_score + language_score
        
        #context = text_context * video_context
        #scores = self.joint_cls(context).squeeze()

        # predict person contained in each bounding box
        char = self.char_classifier(char_prediction_context.unsqueeze(dim=1).repeat(1, video_length, 1))
        

        return scores, char, labels 


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


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

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab, args.n_dim, args.image_dim, args.layers, args.dropout)

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
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, n_head) # dropout? separate attention modules?
        self.linear = nn.Linear(2*hidden_dim, hidden_dim)
        self.relu = nn.ReLU(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm([hidden_dim], elementwise_affine=False)

    def forward(self, target, source_a):
        """Passes the inputs through the utility attention block. For a detailed description see the paper. Inputs are tensors for each utility. The output is the updated utility tensor.
        Args:
            target: the target utility. The output will be of the same shape as this target utility.
            source_a: the first source utility to attend to.
            source_b: the second source utility to attend to.
        """
        # Permute to fit multihead attention input
        target = target.permute(1,0,2)
        source_a = source_a.permute(1,0,2)
        #source_b = source_b.permute(1,0,2)

        # Apply multihead attention mechanism for target and multiple sources as described in the paper
        out_t, _ = self.multihead_attn(target, target, target) # self attention for target utility
        out_a, _ = self.multihead_attn(target, source_a, source_a) # attention to source utility a
        #out_b, _ = self.multihead_attn(target, source_b, source_b) # attention to source utility b

        # Permute back to batch-first
        target = target.permute(1,0,2)
        out_t = out_t.permute(1,0,2)
        out_a = out_a.permute(1,0,2)
        #out_b = out_b.permute(1,0,2)
       
        # Add & norm
        out_a = self.norm(out_a + target)
        #out_b = self.norm(out_b + target)

        #out = torch.cat((out_t, out_a, out_b), dim=2) # concatenate the resulting output tensors
        out = torch.cat([out_t, out_a], dim=2) # concatenate the resulting output tensors
        out = self.relu(self.linear(out)) 
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
    def __init__(self, hidden_dim, feedforward_dim=1024, n_head=5, dropout=0.5):
        super(UtilityLayer, self).__init__()
        self.utility_t = UtilityBlock(hidden_dim, feedforward_dim, n_head, dropout)
        self.utility_v = UtilityBlock(hidden_dim, feedforward_dim, n_head, dropout)
        #self.norm = nn.LayerNorm(hidden_dim)
        #trm_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_head, dim_feedforward=feedforward_dim, dropout=dropout, activation='gelu')
        #self.trm_t = nn.TransformerEncoder(trm_layer, num_layers=1, norm=self.norm)
        #self.trm_v = nn.TransformerEncoder(trm_layer, num_layers=1, norm=self.norm)

    def forward(self, T, V):
        """Passes the input utilities through the utility attention layer. Inputs are passed through their respective blocks in parallel. The output are the three updated utility tensors.
        Args:
            V: the visual utility tensor
            Q: the question utility tensor
            R: the history utility tensor
        """
        T_out = self.utility_t(T, V)
        #T_out = self.trm_t(T_out)
        V_out = self.utility_v(V, T)
        #V_out = self.trm_v(V_out)
        return T_out, V_out
