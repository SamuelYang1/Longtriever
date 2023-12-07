import os
import json
from dataclasses import dataclass
import torch.utils.data.dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorForWholeWordMask
from datasets import load_dataset,concatenate_datasets,load_from_disk
from utils import tensorize_batch
import nltk

nltk.download('punkt')

class DatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, args):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

        cached_path = './cached_data/train'
        if os.path.exists(cached_path):
            print(f'loading dataset from {cached_path}')
            self.dataset = load_from_disk(dataset_path=cached_path)
            return

        def book_tokenize_function(examples):
            return tokenizer(examples["text"], add_special_tokens=False, truncation=False,
                             return_attention_mask=False,return_token_type_ids=False,verbose=False)

        target_length = (args.max_corpus_length - tokenizer.num_special_tokens_to_add(pair=False))*args.max_corpus_sent_num

        def book_pad_each_line(examples):
            texts = []
            blocks = []
            curr_block = []
            for sent in examples['input_ids']:
                if len(curr_block)+len(sent) >= target_length and curr_block:
                    blocks.append(curr_block)
                    curr_block = []
                    if len(blocks)>=args.max_corpus_sent_num:
                        texts.append(blocks)
                        blocks=[]
                curr_block.extend(sent)
            if len(curr_block) > 0:
                blocks.append(curr_block)
            if len(blocks) > 0:
                texts.append(blocks)
            return {'token_ids': texts} # {'token_ids':[[[int]]]]}

        bookcorpus = load_dataset('bookcorpus', split='train')
        tokenized_bookcorpus = bookcorpus.map(book_tokenize_function, num_proc=64, remove_columns=["text"])
        processed_bookcorpus = tokenized_bookcorpus.map(book_pad_each_line, num_proc=64, batched=True,
                                                        batch_size=1000, remove_columns=["input_ids"])

        def wiki_tokenize_function(examples):
            sentences = nltk.sent_tokenize(examples["text"])
            return tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                             return_token_type_ids=False,verbose=False)
            # return {'input_ids':[[int]]}

        def wiki_pad_each_line(examples):
            texts = []
            for sents in examples['input_ids']:
                blocks = []
                curr_block = []
                for sent in sents:
                    if len(curr_block)+len(sent) >= target_length and curr_block:
                        blocks.append(curr_block)
                        curr_block = []
                        if len(blocks)>=args.max_corpus_sent_num:
                            texts.append(blocks)
                            blocks=[]
                    curr_block.extend(sent)
                if len(curr_block) > 0:
                    blocks.append(curr_block)
                if len(blocks) > 0:
                    texts.append(blocks)
            return {'token_ids': texts} # {'token_ids':[[[int]]]]}

        wiki = load_dataset("wikipedia", "20200501.en", split="train")
        wiki = wiki.remove_columns("title")
        tokenized_wiki = wiki.map(wiki_tokenize_function, num_proc=64, remove_columns=["text"])
        processed_wiki = tokenized_wiki.map(wiki_pad_each_line, num_proc=64, batched=True, batch_size=1000,
                                            remove_columns=["input_ids"])

        bert_dataset = concatenate_datasets([processed_bookcorpus, processed_wiki])
        self.dataset = bert_dataset

        self.dataset.save_to_disk(dataset_path=cached_path)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


@dataclass
class LongtrieverCollator(DataCollatorForWholeWordMask):
    max_corpus_sent_num: int = 5
    max_corpus_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):
        encoder_input_ids_batch = []
        encoder_attention_mask_batch = []
        encoder_labels_batch=[]
        decoder_input_ids_batch=[]
        decoder_matrix_attention_mask_batch = []
        decoder_labels_batch=[]

        block_len=self.max_corpus_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:

            input_ids_blocks = []
            attention_mask_blocks = []
            encoder_mlm_mask_blocks=[]
            matrix_attention_mask_blocks=[]
            decoder_labels_blocks=[]

            for token_ids in e['token_ids']:
                input_ids_block = self.tokenizer.build_inputs_with_special_tokens(token_ids[:block_len])
                tokens_block = [self.tokenizer._convert_id_to_token(tid) for tid in input_ids_block]
                self.mlm_probability = self.encoder_mlm_probability
                encoder_mlm_mask_block = self._whole_word_mask(tokens_block)

                self.mlm_probability = self.decoder_mlm_probability
                matrix_attention_mask_block = []
                for i in range(len(tokens_block)):
                    decoder_mlm_mask = self._whole_word_mask(tokens_block)
                    decoder_mlm_mask[i] = 1
                    matrix_attention_mask_block.append(decoder_mlm_mask)

                input_ids_blocks.append(torch.tensor(input_ids_block))
                attention_mask_blocks.append(torch.tensor([1] * len(input_ids_block)))
                input_ids_block[0] = -100
                input_ids_block[-1] = -100
                decoder_labels_blocks.append(torch.tensor(input_ids_block))

                encoder_mlm_mask_blocks.append(torch.tensor(encoder_mlm_mask_block))
                matrix_attention_mask_blocks.append(1 - torch.tensor(matrix_attention_mask_block))

            input_ids_blocks = tensorize_batch(input_ids_blocks, self.tokenizer.pad_token_id)
            attention_mask_blocks = tensorize_batch(attention_mask_blocks, 0)
            origin_input_ids_blocks = input_ids_blocks.clone()
            encoder_mlm_mask_blocks = tensorize_batch(encoder_mlm_mask_blocks, 0)
            encoder_input_ids_blocks, encoder_labels_blocks = self.torch_mask_tokens(input_ids_blocks, encoder_mlm_mask_blocks)
            decoder_labels_blocks = tensorize_batch(decoder_labels_blocks, -100)
            matrix_attention_mask_blocks = tensorize_batch(matrix_attention_mask_blocks, 0)

            encoder_input_ids_batch.append(encoder_input_ids_blocks)
            encoder_attention_mask_batch.append(attention_mask_blocks)
            encoder_labels_batch.append(encoder_labels_blocks)
            decoder_input_ids_batch.append(origin_input_ids_blocks)
            decoder_matrix_attention_mask_batch.append(matrix_attention_mask_blocks)
            decoder_labels_batch.append(decoder_labels_blocks)

        encoder_input_ids_batch=tensorize_batch(encoder_input_ids_batch,self.tokenizer.pad_token_id)
        encoder_attention_mask_batch=tensorize_batch(encoder_attention_mask_batch,0)
        encoder_labels_batch=tensorize_batch(encoder_labels_batch,-100)
        decoder_input_ids_batch=tensorize_batch(decoder_input_ids_batch,self.tokenizer.pad_token_id)
        decoder_matrix_attention_mask_batch=tensorize_batch(decoder_matrix_attention_mask_batch,0)
        decoder_labels_batch=tensorize_batch(decoder_labels_batch,-100)


        batch = {
            "encoder_input_ids_batch": encoder_input_ids_batch,
            "encoder_attention_mask_batch": encoder_attention_mask_batch,
            "encoder_labels_batch": encoder_labels_batch,
            "decoder_input_ids_batch": decoder_input_ids_batch,
            "decoder_matrix_attention_mask_batch": decoder_matrix_attention_mask_batch,  # [B,N,L,L]
            "decoder_labels_batch": decoder_labels_batch,
        }

        return batch


class DatasetForFineTuning(torch.utils.data.Dataset):
    def __init__(self, args):
        def load_jsonl(file_path):
            d={}
            with open(file_path,encoding="utf-8")as df:
                for line in df:
                    query=json.loads(line)
                    d[query['_id']]=query
            return d

        self.id2query=load_jsonl(args.query_file)
        self.id2corpus=load_jsonl(args.corpus_file)
        self.dataset=open(args.qrels_file,encoding="utf-8").readlines()[1:]

    def __getitem__(self, item):
        query_id, corpus_id, score=self.dataset[item].split('\t')
        query_str=self.id2query[query_id].get("text","")
        corpus_title_str=self.id2corpus[corpus_id].get("title","")
        corpus_text_str=self.id2corpus[corpus_id].get("text","")
        corpus_str=corpus_title_str+' '+corpus_text_str if len(corpus_title_str)>0 else corpus_text_str
        return [query_str,corpus_str]

    def __len__(self):
        return len(self.dataset)


@dataclass
class DataCollatorForFineTuningLongtriever:
    tokenizer:PreTrainedTokenizerBase
    max_query_length:int
    max_corpus_length:int
    max_corpus_sent_num:int
    align_right:bool=False
    def __post_init__(self):
        if isinstance(self.tokenizer,str):
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer)
        elif isinstance(self.tokenizer,PreTrainedTokenizerBase):
            pass
        else:
            raise TypeError

    def tokenize(self,string):
        sentences = nltk.sent_tokenize(string)
        if not sentences:
            sentences = ["."]
        results = self.tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                                 return_token_type_ids=False, verbose=False)

        block_len = self.max_corpus_length - self.tokenizer.num_special_tokens_to_add(False)
        input_ids_blocks = []
        attention_mask_blocks = []
        curr_block = []
        for input_ids_sent in results['input_ids']:
            if len(curr_block) + len(input_ids_sent) >= block_len and curr_block:
                input_ids_blocks.append(
                    torch.tensor(self.tokenizer.build_inputs_with_special_tokens(curr_block[:block_len])))
                attention_mask_blocks.append(torch.tensor([1] * len(input_ids_blocks[-1])))
                curr_block = []
                if len(input_ids_blocks) >= self.max_corpus_sent_num:
                    break
            curr_block.extend(input_ids_sent)
        if len(curr_block) > 0:
            input_ids_blocks.append(
                torch.tensor(self.tokenizer.build_inputs_with_special_tokens(curr_block[:block_len])))
            attention_mask_blocks.append(torch.tensor([1] * len(input_ids_blocks[-1])))
        input_ids_blocks = tensorize_batch(input_ids_blocks, self.tokenizer.pad_token_id, align_right=self.align_right)
        attention_mask_blocks = tensorize_batch(attention_mask_blocks, 0, align_right=self.align_right)
        return {
            "input_ids_blocks": input_ids_blocks,
            "attention_mask_blocks": attention_mask_blocks,
        }

    def __call__(self, examples):
        query_input_ids_batch = []
        query_attention_mask_batch = []
        corpus_input_ids_batch = []
        corpus_attention_mask_batch = []
        for e in examples:
            query_str, corpus_str=e

            query_results=self.tokenize(query_str)
            query_input_ids_batch.append(query_results['input_ids_blocks'])
            query_attention_mask_batch.append(query_results['attention_mask_blocks'])

            corpus_resutls=self.tokenize(corpus_str)
            corpus_input_ids_batch.append(corpus_resutls['input_ids_blocks'])
            corpus_attention_mask_batch.append(corpus_resutls['attention_mask_blocks'])

        query_input_ids_batch = tensorize_batch(query_input_ids_batch, self.tokenizer.pad_token_id, align_right=self.align_right)  # [B,N,L]
        query_attention_mask_batch = tensorize_batch(query_attention_mask_batch, 0, align_right=self.align_right)  # [B,N,L]
        corpus_input_ids_batch=tensorize_batch(corpus_input_ids_batch,self.tokenizer.pad_token_id, align_right=self.align_right) #[B,N,L]
        corpus_attention_mask_batch=tensorize_batch(corpus_attention_mask_batch,0, align_right=self.align_right) #[B,N,L]


        batch = {
            "query_input_ids": query_input_ids_batch, #[B,N,L]
            "query_attention_mask": query_attention_mask_batch, #[B,N,L]
            "corpus_input_ids": corpus_input_ids_batch, #[B,N,L]
            "corpus_attention_mask": corpus_attention_mask_batch, #[B,N,L]
        }

        return batch