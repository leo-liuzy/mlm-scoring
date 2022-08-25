from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
ctxs = [mx.cpu()]

model, vocab, tokenizer = get_pretrained(ctxs, 'roberta-base')
scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)
print(scorer.score_sentences(["Hello world!"]))
# >> [-12.411025047302246]
print(scorer.score_sentences(["Hello world!"], per_token=True))