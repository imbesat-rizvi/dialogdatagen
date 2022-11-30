from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer


def sentence_transformers_embeddings(
    sentences,
    batch_size=-1,
    model="sentence-transformers/paraphrase-mpnet-base-v2",
):
    if isinstance(model, str):
        model = SentenceTransformer(model)

    if isinstance(sentences, str):
        sentences = [sentences]

    if batch_size == -1:
        embeddings = model.encode(sentences)
    else:
        embeddings = model.encode(sentences[:batch_size])
        for i in tqdm(range(batch_size, len(sentences), batch_size)):
            batch = model.encode(sentences[i : i + batch_size])
            embeddings = np.vstack((embeddings, batch))

    return embeddings
