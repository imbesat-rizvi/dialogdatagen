import spacy

# from multiprocess import set_start_method
# set_start_method("spawn")

spnlp_model = "en_core_web_sm"
spnlp = spacy.load(spnlp_model)


def spnlp_doc_to_sent(doc, spnlp=spnlp):
    sents = [str(i).strip() for i in spnlp(doc.strip()).sents]
    return sents


def hf_doc_to_sent(
    ex,
    text_col,
    doc_to_sent_fn=spnlp_doc_to_sent,
    sent_col="sent",
):
    other_cols = [i for i in ex.keys() if i != text_col]
    sents = {c: [] for c in other_cols}
    sents[sent_col] = []

    for i, doc in enumerate(ex[text_col]):
        sent = doc_to_sent_fn(doc)
        sents[sent_col] += sent

        for c in other_cols:
            sents[c] += [ex[c][i]] * len(sent)

    return sents
