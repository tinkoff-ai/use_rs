"""
SHAN
################################################

Reference:
    Ying, H et al. "Sequential Recommender System based on Hierarchical Attention Network."in IJCAI 2018


"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_, uniform_

from models.BaseModel import SequentialModel

class SHAN2(SequentialModel):
    """
    SHAN exploit the Hierarchical Attention Network to get the long-short term preference
    first get the long term purpose and then fuse the long-term with recent items to get long-short term purpose

    """   
    extra_log_args = ['emb_size', 'short_item_length', 'reg_weight']
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--short_item_length', type=int, default=2,
                            help='The last N items.')
        parser.add_argument('--reg_weight', type=float, default=0.001,
                        help='The L2 regularization weight.')
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.short_item_length = args.short_item_length
        self.max_seq_length = args.history_max
        assert self.short_item_length <= self.max_seq_length, "short_item_length can't longer than the max_seq_length"
        self.reg_weight = args.reg_weight
        super().__init__(args, corpus)

    def _define_params(self):
        # define layers and loss
        self.item_embedding = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        self.user_embedding = nn.Embedding(self.user_num, self.emb_size)

        self.long_w = nn.Linear(self.emb_size, self.emb_size)
        self.long_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.emb_size),
                a=-np.sqrt(3 / self.emb_size),
                b=np.sqrt(3 / self.emb_size)
            ),
            requires_grad=True
        ).to(self.device)
        self.long_short_w = nn.Linear(self.emb_size, self.emb_size)
        self.long_short_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.emb_size),
                a=-np.sqrt(3 / self.emb_size),
                b=np.sqrt(3 / self.emb_size)
            ),
            requires_grad=True
        ).to(self.device)

        self.relu = nn.ReLU()

        self.emb_dropout = nn.Dropout(0.1)
        self.gru_layers = nn.GRU(
            input_size=self.emb_size,
            hidden_size=self.emb_size*2,
            num_layers=2,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.emb_size*2, self.emb_size)
        # init the parameters of the model
        self.apply(self.init_weights)

    def inverse_seq_item(self, seq_item, seq_item_len):
        """
        inverse the seq_item, like this
            [1,2,3,0,0,0,0] -- after inverse -->> [0,0,0,0,1,2,3]
        """
        seq_item = seq_item.cpu().numpy()
        seq_item_len = seq_item_len.cpu().numpy()
        new_seq_item = []
        for items, length in zip(seq_item, seq_item_len):
            item = list(items[:length])
            zeros = list(items[length:])
            seqs = zeros + item
            new_seq_item.append(seqs)
        seq_item = torch.tensor(new_seq_item, dtype=torch.long, device=self.device)

        return seq_item

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0., 0.01)
        elif isinstance(module, nn.Linear):
            uniform_(module.weight.data, -np.sqrt(3 / self.emb_size), np.sqrt(3 / self.emb_size))
        elif isinstance(module, nn.Parameter):
            uniform_(module.data, -np.sqrt(3 / self.emb_size), np.sqrt(3 / self.emb_size))
            print(module.data)

    def forward(self, feed_dict):
        self.check_list = []
        seq_item = feed_dict['history_items']
        user = feed_dict['user_id']
        item = feed_dict['item_id']
        seq_item_len  = feed_dict['lengths']
        seq_item = self.inverse_seq_item(seq_item, seq_item_len)

        seq_item_embedding = self.item_embedding(seq_item)
 
        user_seq_emb = self.user_embedding(seq_item)
        user_seq_emb_dropout = self.emb_dropout(user_seq_emb)
        gru_output, user_embedding  = self.gru_layers(user_seq_emb_dropout)
        user_embedding = user_embedding [1]
        user_embedding = self.dense(user_embedding)

        # get the mask
        mask = seq_item.data.eq(0)
        
        long_term_attention_based_pooling_layer = self.long_term_attention_based_pooling_layer(
            seq_item_embedding, user_embedding, mask
        )
        # batch_size * 1 * embedding_size

        short_item_embedding = seq_item_embedding[:, -self.short_item_length:, :]
        mask_long_short = mask[:, -self.short_item_length:]
        batch_size = mask_long_short.size(0)
        x = torch.zeros(size=(batch_size, 1)).eq(1).to(self.device)
        mask_long_short = torch.cat([x, mask_long_short], dim=1)
        # batch_size * short_item_length * embedding_size
        long_short_item_embedding = torch.cat([long_term_attention_based_pooling_layer, short_item_embedding], dim=1)
        # batch_size * 1_plus_short_item_length * embedding_size

        long_short_item_embedding = self.long_and_short_term_attention_based_pooling_layer(
            long_short_item_embedding, user_embedding, mask_long_short
        )
        # batch_size * embedding_size
        i_vectors = self.item_embedding(item)
        prediction = (long_short_item_embedding[:, None, :] * i_vectors).sum(-1)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
 
    def long_and_short_term_attention_based_pooling_layer(self, long_short_item_embedding, user_embedding, mask=None):
        """

        fusing the long term purpose with the short-term preference
        """
        long_short_item_embedding_value = long_short_item_embedding

        long_short_item_embedding = self.relu(self.long_short_w(long_short_item_embedding) + self.long_short_b)
        long_short_item_embedding = torch.matmul(long_short_item_embedding, user_embedding.unsqueeze(2)).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            long_short_item_embedding.masked_fill_(mask, -1e9)
        long_short_item_embedding = nn.Softmax(dim=-1)(long_short_item_embedding)
        long_short_item_embedding = torch.mul(long_short_item_embedding_value,
                                              long_short_item_embedding.unsqueeze(2)).sum(dim=1)

        return long_short_item_embedding

    def long_term_attention_based_pooling_layer(self, seq_item_embedding, user_embedding, mask=None):
        """

        get the long term purpose of user
        """
        seq_item_embedding_value = seq_item_embedding

        seq_item_embedding = self.relu(self.long_w(seq_item_embedding) + self.long_b)
        user_item_embedding = torch.matmul(seq_item_embedding, user_embedding.unsqueeze(2)).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            user_item_embedding.masked_fill_(mask, -1e9)
        user_item_embedding = nn.Softmax(dim=1)(user_item_embedding)
        user_item_embedding = torch.mul(seq_item_embedding_value,
                                        user_item_embedding.unsqueeze(2)).sum(dim=1, keepdim=True)
        # batch_size * 1 * embedding_size

        return user_item_embedding
