import logging
import os
import sys
import datasets

from arguments import DataTrainingArguments, ModelArguments
from trainer import PreTrainer
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,TrainingArguments,
    )
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    training_args.remove_unused_columns=False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    datasets.set_progress_bar_enabled(False)

    # Create datacollator & model
    tokenizer=AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    if model_args.model_type=="longtriever":
        from data_handler import DatasetForFineTuning,DataCollatorForFineTuningLongtriever
        dataset = DatasetForFineTuning(data_args)
        data_collator=DataCollatorForFineTuningLongtriever(tokenizer,data_args.max_query_length,data_args.max_corpus_length,data_args.max_corpus_sent_num)
    elif model_args.model_type=="longtriever_pretrain":
        from data_handler import DatasetForPretraining,LongtrieverCollator
        dataset = DatasetForPretraining(data_args)
        data_collator=LongtrieverCollator(tokenizer,max_corpus_sent_num=data_args.max_corpus_sent_num,max_corpus_length=data_args.max_corpus_length,
                                           encoder_mlm_probability=data_args.encoder_mlm_probability, decoder_mlm_probability=data_args.decoder_mlm_probability)
    else:
        raise ValueError

    if model_args.model_type=="longtriever":
        from modeling_longtriever import Longtriever
        from modeling_retriever import LongtrieverRetriever
        model = LongtrieverRetriever(Longtriever.from_pretrained(model_args.model_name_or_path))
    elif model_args.model_type=="longtriever_pretrain":
        from modeling_longtriever import LongtrieverForPretraining
        model = LongtrieverForPretraining.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError

    # Initialize our Trainer
    trainer = PreTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

if __name__ == "__main__":
    main()