"""
HGN
################################################

Reference:
    Chen Ma et al. "Hierarchical Gating Networks for Sequential Recommendation."in SIGKDD 2019


"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_, normal_
import torch.nn.functional as F
from models.BaseModel import SequentialModel

class HGN2(SequentialModel):
    """
    HGN sets feature gating and instance gating to get the important feature and item for predicting the next item

    """
    extra_log_args = ['emb_size', 'reg_weight', 'pool_type']
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--reg_weight', type=float, default=0.001,
                            help='The L2 regularization weight.')
        parser.add_argument('--pool_type', type=str, default='average',
                            help='The type of pooling.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.reg_weight = args.reg_weight
        self.pool_type = args.pool_type
        self.max_seq_length = args.history_max
        if self.pool_type not in ["max", "average"]:
            raise NotImplementedError("Make sure 'pool_type' in ['max', 'average']!")
        super().__init__(args, corpus)

    def _define_params(self):
        # define the layers and loss function
        self.item_embedding = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        self.user_embedding = nn.Embedding(self.user_num, self.emb_size)

        # define the module feature gating need
        self.w1 = nn.Linear(self.emb_size, self.emb_size)
        self.w2 = nn.Linear(self.emb_size, self.emb_size)
        self.b = nn.Parameter(torch.zeros(self.emb_size), requires_grad=True).to(self.device)

        # define the module instance gating need
        self.w3 = nn.Linear(self.emb_size, 1, bias=False)
        self.w4 = nn.Linear(self.emb_size, self.max_seq_length, bias=False)

        # define item_embedding for prediction
        self.item_embedding_for_prediction = nn.Embedding(self.item_num, self.emb_size)

        self.sigmoid = nn.Sigmoid()

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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0., 1 / self.emb_size)
        elif isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def feature_gating(self, seq_item_embedding, user_embedding):
        """
        choose the features that will be sent to the next stage(more important feature, more focus)
        """
        batch_size, seq_len, embedding_size = seq_item_embedding.size()
        seq_item_embedding_value = seq_item_embedding

        seq_item_embedding = self.w1(seq_item_embedding)
        # batch_size * seq_len * embedding_size
        user_embedding = self.w2(user_embedding)
        # batch_size * embedding_size
        user_embedding = user_embedding.unsqueeze(1).repeat(1, seq_len, 1)
        # batch_size * seq_len * embedding_size

        user_item = self.sigmoid(seq_item_embedding + user_embedding + self.b)
        # batch_size * seq_len * embedding_size

        user_item = torch.mul(seq_item_embedding_value, user_item)
        # batch_size * seq_len * embedding_size

        return user_item

    def instance_gating(self, user_item, user_embedding):
        """
        choose the last click items that will influence the prediction( more important more chance to get attention)
        """
        user_embedding_value = user_item
        user_item = self.w3(user_item)
        # batch_size * seq_len * 1
        user_embedding = self.w4(user_embedding).unsqueeze(2)
        # batch_size * seq_len * 1
        instance_score = self.sigmoid(user_item + user_embedding).squeeze(-1)
        # batch_size * seq_len * 1
        output = torch.mul(instance_score.unsqueeze(2), user_embedding_value)
        # batch_size * seq_len * embedding_size

        if self.pool_type == "average":
            output = torch.div(output.sum(dim=1), instance_score.sum(dim=1).unsqueeze(1))
            # batch_size * embedding_size
        else:
            # for max_pooling
            index = torch.max(instance_score, dim=1)[1]
            # batch_size * 1
            output = self.gather_indexes(output, index)
            # batch_size * seq_len * embedding_size ==>> batch_size * embedding_size

        return output

    def forward(self, feed_dict):
        self.check_list = []
        seq_item = feed_dict['history_items'] 
        user = feed_dict['user_id']
        item = feed_dict['item_id']
        batch_size, seq_len = seq_item.shape
        pad_len = self.max_seq_length - seq_len
        seq_item = F.pad(seq_item, [0, pad_len])
        seq_item_embedding = self.item_embedding(seq_item)

        user_seq_emb = self.user_embedding(seq_item)
        user_seq_emb_dropout = self.emb_dropout(user_seq_emb)
        gru_output, user_embedding  = self.gru_layers(user_seq_emb_dropout)
        user_embedding = user_embedding [1]
        user_embedding = self.dense(user_embedding)

        feature_gating = self.feature_gating(seq_item_embedding, user_embedding)
        instance_gating = self.instance_gating(feature_gating, user_embedding)
        # batch_size * embedding_size
        item_item = torch.sum(seq_item_embedding, dim=1)
        # batch_size * embedding_size
        prediction = user_embedding + instance_gating + item_item
        i_vectors = self.item_embedding(item)
        prediction = (prediction[:, None, :] * i_vectors).sum(-1)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
