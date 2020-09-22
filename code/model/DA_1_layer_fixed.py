import torch
from torch import nn
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Encoders.DAEncoder import DAEncoder
from model.Decoders.Disc import Disc

class DA_1_layer_fixed(nn.Module):
    """Convenience wrapper module, wrapping Encoder and Decoder modules.
    Parameters
    ----------
    encoder: nn.Module
    decoder: nn.Module
    """

    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()
        print("Model name: DA, 1 layer, fixed subspaces")
        self.vocab = vocab
        self.encoder = Encoder(args, vocab, n_dim, image_dim, layers, dropout, num_choice).cuda()
        self.decoder = Decoder(args, vocab, n_dim, image_dim, layers, dropout, num_choice).cuda()

    def forward(self, que, answers, **features):
        encoder_output = self.encoder(que, answers, **features)
        decoder_output = self.decoder(encoder_output, que, answers, **features)
        return decoder_output

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab, args.n_dim, args.image_dim, args.layers, args.dropout)

    def load_embedding(self, pretrained_embedding):
        print('Load pretrained embedding ...')
        self.encoder.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding)) 
