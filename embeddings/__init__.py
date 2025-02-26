from .time_embedding import *
from .type_embedding import *


# EMBEDDING_DICT = {
#     'Trigo': TrigonoTimeEmbedding
# }
#
# def get_embedding(**args):
#     time_emb_name = args['time_emb']
#     return EMBEDDING_DICT[time_emb_name](embed_size=args['embed_size'] // 2),\
#            TypeEmbedding(embed_size = args['embed_size'] // 2, padding_idx=args['event_type_num'], event_type_num=args['event_type_num'])

def get_embedding(**args):
    time_emb_name = args['time_emb']
    return TrigonoTimeEmbedding(embed_size=args['embed_size'] // 2),\
           TypeEmbedding(embed_size = args['embed_size'] // 2, padding_idx=args['event_type_num'], event_type_num=args['event_type_num'])
