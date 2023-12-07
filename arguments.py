from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataTrainingArguments:
    tokenizer_name: Optional[str] = field(default="bert-base-uncased")
    query_file: Optional[str] = field(default="./msmarco_doc_small/queries.jsonl")
    corpus_file: Optional[str] = field(default="./msmarco_doc_small/corpus.jsonl")
    qrels_file: Optional[str] = field(default="./msmarco_doc_small/qrels/train.tsv")
    max_query_length: Optional[int] = field(default=512)
    max_corpus_length: Optional[int] = field(default=512)
    max_corpus_sent_num: Optional[int] = field(default=5)
    encoder_mlm_probability: Optional[float] = field(default=0.3)
    decoder_mlm_probability: Optional[float] = field(default=0.5)

@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="bert")
    model_name_or_path: Optional[str] = field(default="bert-base-uncased")