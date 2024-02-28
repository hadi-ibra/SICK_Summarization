import os
from config.args import get_parser
import argparse
import nltk
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
import wandb
from data.dataset import SamsumDataset_total, DialogsumDataset_total

nltk.download("punkt")


os.environ["WANDB_SILENT"] = "true"

MY_TOKEN = "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"


def get_args() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
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


def gen_prompt(training_example: str, training_summary: str, test_example: str) -> str:
    template = f"""
    Summarize the chat dialog.
    Here you can find some examples:
    DIALOG:
    ```{training_example}```
    SUMMARY:
    ```{training_summary}```
    Summarize the following chat dialog:
    DIALOG:
    ```{test_example}```
    SUMMARY:
    """
    return template


def main():
    try:
        args = get_args()
        device = is_cuda_available()
        set_wandb(args)

        model_tag = "meta-llama/Llama-2-7b-chat-hf"

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_tag, use_auth_token=MY_TOKEN)
        # This line is for debug we have no idea why we should use it
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        special_tokens_dict = {"additional_special_tokens": ["<I>", "</I>"]}
        tokenizer.add_special_tokens(special_tokens_dict)

        # Model
        # model = AutoModelForCausalLM.from_pretrained(
        # model_tag, use_auth_token=MY_TOKEN, torch_dtype=torch.float16, device_map=device
        # )
        model = AutoModelForCausalLM.from_pretrained(model_tag, use_auth_token=MY_TOKEN, device_map=device)

        # Dataset
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
            is_llm=True,
        )
        train_dataset = torch.utils.data.Subset(total_dataset.getTrainData(), [i for i in range(10)])
        eval_dataset = torch.utils.data.Subset(total_dataset.getEvalData(), [i for i in range(5)])
        test_dataset = torch.utils.data.Subset(total_dataset.getTestData(), [i for i in range(5)])

        # Prompt

        temperature = 0
        length_max = 2048

        model.resize_token_embeddings(len(tokenizer))

        while True:
            training_example, training_summary = train_dataset.__getitem__(0)
            test_example, test_summary = eval_dataset.__getitem__(0)
            prompt = gen_prompt(training_example, training_summary, test_example)
            inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
            generate_ids = model.generate(
                **inputs,
                # do_sample=True if temperature > 0 else False,
                do_sample=False,
                temperature=temperature,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=length_max,
            )
            # text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(text)
            input("Press a key to continue")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
