from datasets import load_dataset
from functools import partial
import multiprocessing as mp

from .utils import spnlp_doc_to_sent, hf_doc_to_sent


class MultiDoc2Dial:
    NAME = "multidoc2dial"
    CONFIG = "document_domain"
    REMOVE_COLS = ["spans", "doc_html_ts", "doc_html_raw"]

    def __init__(
        self,
        name=NAME,
        config=CONFIG,
        split="train",
        remove_cols=REMOVE_COLS,
    ):
        self.doc = load_dataset(name, name=config, split=split)
        if remove_cols:
            self.doc = self.doc.remove_columns(remove_cols)

    def doc_to_sent(
        self,
        doc_to_sent_fn=spnlp_doc_to_sent,
        text_col="doc_text",
        sent_col="sent",
        batched=True,
        num_proc=mp.cpu_count(),
    ):
        self.doc = self.doc.map(
            partial(
                hf_doc_to_sent,
                text_col=text_col,
                doc_to_sent_fn=doc_to_sent_fn,
                sent_col=sent_col,
            ),
            batched=batched,
            remove_columns=self.doc.column_names,
            num_proc=num_proc,
        )

        return self
