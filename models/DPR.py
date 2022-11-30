from tqdm import tqdm
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRReader, DPRReaderTokenizerFast

QUESTION_ENCODER = "facebook/dpr-question_encoder-single-nq-base"
CONTEXT_ENCODER = "facebook/dpr-ctx_encoder-single-nq-base"
READER = "facebook/dpr-reader-single-nq-base"


class DPR:
    def __init__(
        self,
        query_encoder=QUESTION_ENCODER,
        ctx_encoder=CONTEXT_ENCODER,
        reader=READER,
    ):
        self.query_tokenizer, self.ctx_tokenizer, self.reader_tokenizer = (
            None,
            None,
            None,
        )
        self.query_encoder, self.ctx_encoder, self.reader = None, None, None

        if query_encoder:
            self.query_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(
                query_encoder
            )
            self.query_encoder = DPRQuestionEncoder.from_pretrained(query_encoder)

        if ctx_encoder:
            self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
                ctx_encoder
            )
            self.ctx_encoder = DPRContextEncoder.from_pretrained(ctx_encoder)

        if reader:
            self.reader_tokenizer = DPRReaderTokenizerFast.from_pretrained(reader)
            self.reader = DPRReader.from_pretrained(reader)

    def _get_embs(self, texts, tokenizer, encoder):
        input_ids = tokenizer(texts, return_tensors="pt")["input_ids"]
        embeddings = encoder(input_ids).pooler_output[0]
        return embeddings

    def dpr_embeddings(
        self,
        sentences,
        batch_size=-1,
        emb_model="query",
    ):
        if isinstance(sentences, str):
            sentences = [sentences]

        if emb_model == "query":
            tokenizer = self.query_tokenizer
            encoder = self.query_encoder
        else:
            tokenizer = self.ctx_tokenizer
            encoder = self.ctx_encoder

        if batch_size == -1:
            embeddings = self._get_embs(sentences, tokenizer, encoder)
        else:
            embeddings = self._get_embs(sentences[:batch_size], tokenizer, encoder)
            for i in tqdm(range(batch_size, len(sentences), batch_size)):
                batch = self._get_embs(
                    sentences[i : i + batch_size], tokenizer, encoder
                )
                embeddings = torch.vstack((embeddings, batch))

        return embeddings
