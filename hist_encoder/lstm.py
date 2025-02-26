from torch import nn
from .encoder import Encoder

class LSTMEnc(Encoder):
    def __init__(
        self,
        event_type_num: int,
        embed_size: int = 16,
        input_size: int = 32,
        layer_num: int =1,
        drop_ratio: float = 0.1,
        activation = 'tanh',
        **kwargs
    ):
        super().__init__(event_type_num, embed_size,input_size, layer_num, drop_ratio, activation)
        self.recurrent_nn = nn.LSTM(input_size = embed_size, hidden_size = embed_size, num_layers = layer_num, batch_first = True)

    def _hist_encoding(self, seq_types, embedding, lag_matrixes = None, similarity_matrixes=None):
        # batch_size, seq_len, emb_size = embedding.shape
        history_embedding, _ = self.recurrent_nn(embedding)
        return history_embedding

    def forward(self, seq_types, embeddings, la_matrix = None, similarity_matrixes = None):
        return self._hist_encoding(seq_types, embeddings, la_matrix, similarity_matrixes)