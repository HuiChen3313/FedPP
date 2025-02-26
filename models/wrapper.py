import torch
import torch.nn as nn
import math

class wrapper(nn.Module):

    def __init__(
        self,
        time_embedding: nn.Module,
        type_embedding: nn.Module,
        encoder: nn.Module,
        event_type_num: int,
        **kwargs
    ):

        super().__init__()
        self.time_emb = time_embedding
        self.type_emb = type_embedding
        self.encoder = encoder
        self.event_type_num = event_type_num


    def _transform_dt(self, seq_dts):
        """
        Convert seqence of time intervals into normalized vector.

        Args:
            seq_dts: Time intervals of events (batch_size, ... ,seq_len)

        Returns:
            seq_dts: Normalized time intervals (batch_size, ... ,seq_len)

        """
        seq_dts = torch.log(seq_dts + 1e-8)
        seq_dts = torch.divide((seq_dts - self.mean_log_inter_time), self.std_log_inter_time)
        return seq_dts


    def _event_embedding(self, seq_dts, seq_types):
        """
        Calculate the embedding from the sequence of events.

        Args:
            seq_dts: Time interval of events (batch_size, ... ,seq_len)
            seq_types: Sequence of event types (batch_size, ... ,seq_len)
        Returns:
            embedding: The embedding of time and event types (batch_size, ..., seq_len, embed_size)

        """
        seq_dts = self._transform_dt(seq_dts)

        type_embedding = self.type_emb(seq_types) * math.sqrt(self.embed_size//2)
        time_embedding = self.time_emb(seq_dts)

        embedding = torch.cat([time_embedding, type_embedding], dim=-1)
        return embedding

    def forward(self, seq_dts, seq_types, lag_matrixes=None, *args):

        event_embedding = self._event_embedding(seq_dts, seq_types)
        similarity_matrixes = self._compute_similarity(seq_dts, seq_types)

        # self.history_embedding = self.encoder(seq_types, event_embedding, lag_matrixes, similarity_matrixes)

        # if self.type_wise == True:
        #     self.history_embedding = self.history_embedding.unsqueeze(dim=-2).expand(-1, -1, self.event_type_num, -1)
        # else:
        #     self.history_embedding = self.history_embedding.unsqueeze(dim=-2)

        return self.history_embedding

