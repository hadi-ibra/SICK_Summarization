import os
from config.args import get_parser
import argparse
import nltk
import numpy as np

import torch
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
# import wandb
from dataset import SamsumDataset_total, DialogsumDataset_total

import nltk
import yaml

nltk.download("punkt")


os.environ["WANDB_SILENT"] = "true"


with open('config/secret.yaml', 'r') as file:
    config_data = yaml.safe_load(file)



MY_TOKEN = config_data['WANDB_API_KEY']
# MY_TOKEN = "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"


def get_args() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    # set_seed(args.seed)
    return args


def is_cuda_available() -> torch.device:
    print("######################################################################")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("Current cuda device:", torch.cuda.current_device())
        print("Count of using GPUs:", torch.cuda.device_count())
        print(torch.cuda.get_device_name())
    print("######################################################################")
    return device


def set_wandb(args: argparse.Namespace) -> None:
    if args.use_paracomet:
        cs = "para"
        if args.use_roberta:
            cs += "_roberta"
    else:
        cs = "comet"
        if args.use_roberta:
            cs += "_roberta"

    if args.use_sentence_transformer:
        if args.use_paracomet:
            cs = "paracomet_sentence_transformer"
        else:
            cs = "comet_sentence_transformer"

    print("#" * 50)
    print(cs)
    print("#" * 50)

    run_name = f"context_{args.dataset_name}_{args.relation}_{cs}_lr{str(args.init_lr)}"

    wandb.init(project="basic_training", reinit=True, entity="nlp_sick_polito", name=run_name)


def a(args: argparse.Namespace) -> None:
    # Define Global Values
    tokenizer_list = {
        "facebook/bart-large": "RobertaTokenizer",
        "facebook/bart-large-xsum": "RobertaTokenizer",
        "google/pegasus-large": "PegasusTokenizer",
        "google/peagsus-xsum": "PegasusTokenizer",
        "google/t5-large-lm-adapt": "T5Tokenizer",
        "google/t5-v1_1-large": "T5Tokenizer",
    }
    max_len_list = {
        "facebook/bart-large": 1024,
        "facebook/bart-large-xsum": 1024,
        "google/pegasus-large": 1024,
        "google/peagsus-xsum": 512,
        "google/t5-large-lm-adapt": 512,
        "google/t5-v1_1-large": 512,
    }
    vocab_size_list = {
        "facebook/bart-large": 50265,
        "facebook/bart-large-xsum": 50264,
        "google/pegasus-large": 96103,
        "google/peagsus-xsum": 96103,
        "google/t5-large-lm-adapt": 32128,
        "google/t5-v1_1-large": 32128,
    }

    # (TODO: this is disgusting, refactor it.) Refine arguments based on global values
    args.tokenizer_name = tokenizer_list[args.model_name]
    args.vocab_size = vocab_size_list[args.model_name]


def get_metric():
    # Set metric
    # metric = load_metric("rouge")
    # return load_metric("../utils/rouge.py")
    return load_metric("src/utils/rouge.py")


def get_tokenizer(model_name: str):
    # Load Tokenizer associated to the model
    # tokenizer = AutoTokenizer.from_pretrained(model_name, token=MY_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add special token
    special_tokens_dict = {"additional_special_tokens": ["<I>", "</I>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def get_dataset(args: argparse.Namespace):
    if args.dataset_name == "samsum":
        total_dataset = SamsumDataset_total(
            args.encoder_max_len,
            args.decoder_max_len,
            tokenizer,
            extra_context=True,
            paracomet=args.use_paracomet,
            relation=args.relation,
            supervision_relation=args.supervision_relation,
            roberta=args.use_roberta,
            sentence_transformer=args.use_sentence_transformer,
            idiom = args.idiom
        )
        train_dataset = total_dataset.getTrainData()
        eval_dataset = total_dataset.getEvalData()
        test_dataset = total_dataset.getTestData()
    elif args.dataset_name == "dialogsum":
        total_dataset = DialogsumDataset_total(
            args.encoder_max_len,
            args.decoder_max_len,
            tokenizer,
            extra_context=True,
            paracomet=args.use_paracomet,
            relation=args.relation,
            supervision_relation=args.supervision_relation,
            sentence_transformer=args.use_sentence_transformer,
            roberta=args.use_roberta,
            idiom = args.idiom
        )
        train_dataset = total_dataset.getTrainData()
        eval_dataset = total_dataset.getEvalData()
        test_dataset = total_dataset.getTestData()
    elif args.dataset_name == "samsum_debug":
        total_dataset = SamsumDataset_total(
            args.encoder_max_len,
            args.decoder_max_len,
            tokenizer,
            extra_context=True,
            paracomet=args.use_paracomet,
            relation=args.relation,
            supervision_relation=args.supervision_relation,
            roberta=args.use_roberta,
            sentence_transformer=args.use_sentence_transformer,
            idiom = args.idiom
        )
        train_dataset = torch.utils.data.Subset(total_dataset.getTrainData(), [i for i in range(10)])
        eval_dataset = torch.utils.data.Subset(total_dataset.getEvalData(), [i for i in range(5)])
        test_dataset = torch.utils.data.Subset(total_dataset.getTestData(), [i for i in range(5)])
    print("######################################################################")
    print("Training Dataset Size is : ")
    print(len(train_dataset))
    print("Validation Dataset Size is : ")
    print(len(eval_dataset))
    print("Test Dataset Size is : ")
    print(len(test_dataset))
    print("######################################################################")
    return total_dataset, train_dataset, eval_dataset, test_dataset


def get_checkpoint(model_name: str, tokenizer, device: torch.device):
    # Loading checkpoint of model
    config = AutoConfig.from_pretrained(model_name, token=MY_TOKEN)
    finetune_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("######################################################################")
    print("Number of Model Parameters are : ", finetune_model.num_parameters())
    print("######################################################################")

    # Set extra Configuration for Finetuning on Summarization Dataset
    finetune_model.resize_token_embeddings(len(tokenizer))
    finetune_model.gradient_checkpointing_enable()
    finetune_model = finetune_model.to(device)
    return finetune_model


def freeze_weight(model):
    for param in model.parameters():
        param.requires_grad = False


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def preprocess_logits_for_metrics(logits, labels):
    print(logits.size())
    logits, _ = logits
    logits_device = logits.device
    logits_reduced = np.argmax(logits.cpu(), axis=-1)
    logits_reduced = logits_reduced.to(logits_device)

    return logits_reduced


if __name__ == "__main__":
    try:
        args = get_args()
        args.model_name = "facebook/bart-large-xsum"
        device = is_cuda_available()
        set_wandb(args)
        a(args)
        metric = get_metric()
        tokenizer = get_tokenizer(args.model_name)
        total_dataset, train_dataset, eval_dataset, test_dataset = get_dataset(args)
        finetune_model = get_checkpoint(args.model_name, tokenizer, device)

        if args.freeze_encoder:
            freeze_weight(finetune_model.get_encoder())

        # Set Training Arguments (& Connect to WANDB)
        finetune_args = Seq2SeqTrainingArguments(
            output_dir=args.finetune_weight_path,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=True,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            # eval_steps=1,
            # logging_steps=1,
            # save_steps=1,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            learning_rate=args.init_lr,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_eps,
            num_train_epochs=args.epoch,
            max_grad_norm=0.1,
            # label_smoothing_factor=0.1,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            # max_steps= ,
            lr_scheduler_type="polynomial",
            # warmup_ratio= ,
            warmup_steps=args.warm_up,
            save_total_limit=1,
            # fp16=True,
            seed=516,
            load_best_model_at_end=True,
            predict_with_generate=True,
            prediction_loss_only=False,
            generation_max_length=100,
            generation_num_beams=5,
            metric_for_best_model="eval_rouge1",
            greater_is_better=True,
            report_to="wandb",
        )

        finetune_trainer = Seq2SeqTrainer(
            model=finetune_model,
            args=finetune_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        # Run Training (Finetuning)
        finetune_trainer.train()

        # Save final weights
        finetune_trainer.save_model(args.best_finetune_weight_path)

    finally:
        wandb.finish()


"""
# Run Evaluation on Test Data
results = finetune_trainer.predict(
    test_dataset=test_dataset,
    max_length= 60,
    num_beams = 5   #1,3,5,10
)
predictions, labels, metrics = results
print('######################################################################')
print("Final Rouge Results are : ",metrics)
print('######################################################################')


# Write evaluation predictions on txt file
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after e ach sentence
decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
decoded_labels = [" ".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]


# output summaries on test set
with open(args.test_output_file_name,"w") as f: 
    f.write(metrics)
    for i in decoded_preds:
        f.write(i.replace("\n","")+"\n")
"""
