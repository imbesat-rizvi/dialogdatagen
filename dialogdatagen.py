from data import MultiDoc2Dial
from models.sentence_transformers import sentence_transformers_embeddings
from models import DPR
from index import Annoy

md2d = MultiDoc2Dial().doc_to_sent()
# st_embs = sentence_transformers_embeddings(md2d.doc["sent"]) #, batch_size=1000)

md2d_mpnet_index_path = "output/index/md2d_sent_mpnet.index"
md2d_mpnet_index = Annoy().load(md2d_mpnet_index_path)

md2d_dpr_query_index_path = "output/index/md2d_sent_dpr_query.index"
md2d_dpr_query_index = Annoy().load(md2d_dpr_query_index_path)

md2d_dpr_ctx_index_path = "output/index/md2d_sent_dpr_ctx.index"
md2d_dpr_ctx_index = Annoy().load(md2d_dpr_ctx_index_path)

i_mpnet, d_mpnet = md2d_mpnet_index.get_knn(
    queries=[0], k=6, search_k=-1, query_as_vector=False
)
i_dpr_query, d_dpr_query = md2d_dpr_query_index.get_knn(
    queries=[0], k=6, search_k=-1, query_as_vector=False
)
i_dpr_ctx, d_dpr_ctx = md2d_dpr_ctx_index.get_knn(
    queries=[0], k=6, search_k=-1, query_as_vector=False
)
i_dpr_qc, d_dpr_qc = md2d_dpr_ctx_index.get_knn(
    queries=[md2d_dpr_query_index[0]], k=6, search_k=-1
)
