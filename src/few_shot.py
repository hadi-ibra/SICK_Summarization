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
import json
from rouge import Rouge
from bert_score import score

nltk.download("punkt")


os.environ["WANDB_SILENT"] = "true"

MY_TOKEN = "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"

SEED = 42


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


def gen_examples_template(training_examples: str) -> str:
    header = "Summarize the chat dialog. Here you can find some examples:"
    tail = "Summarize the following chat dialog in one sentence.\nDIALOG:"
    examples = []
    for dialog, summary in training_examples:
        template_example = f"DIALOG: {dialog}\nSUMMARY: {summary}"
        examples.append(template_example)
    return header + " ".join(examples) + tail


def get_examples(trainds, num_examples=2):
    gen = np.random.default_rng(SEED)
    idxes = gen.integers(0, len(trainds), size=num_examples)
    examples = []
    for idx in idxes:
        dialog, summary = trainds[idx]
        examples.append((dialog, summary))
    return examples


def save_to_json(data, filename):
    with open(f"{filename}.json", "w") as f:
        f.write(json.dumps(data))


def compute_metrics(summaries, device) -> dict:
    rouge = Rouge()
    model_summaries, gold_summaries = map(list, zip(*[s for s in summaries]))
    score_tot = rouge.get_scores(model_summaries, gold_summaries, avg=True)
    p_b, r_b, f_b = score(
        model_summaries,
        gold_summaries,
        lang="en",
        model_type="microsoft/deberta-large-mnli",
        batch_size=1,
        device=device,
    )
    p_bert = p_b.mean()
    r_bert = r_b.mean()
    f_bert = f_b.mean()

    score_tot["bert"] = {"r": r_bert, "p": p_bert, "f": f_bert}

    return score_tot


def main():
    try:
        args = get_args()
        device = is_cuda_available()
        #set_wandb(args)

        model_tag = "meta-llama/Llama-2-7b-chat-hf"

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_tag, use_auth_token=MY_TOKEN)
        # This line is for debug we have no idea why we should use it
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        special_tokens_dict = {"additional_special_tokens": ["<I>", "</I>"]}
        tokenizer.add_special_tokens(special_tokens_dict)

        # Model
        model = AutoModelForCausalLM.from_pretrained(
            model_tag, use_auth_token=MY_TOKEN, torch_dtype=torch.float16, device_map=device
        )
        # model = AutoModelForCausalLM.from_pretrained(model_tag, use_auth_token=MY_TOKEN, device_map=device)

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
        # train_dataset = torch.utils.data.Subset(total_dataset.getTrainData(), [i for i in range(10)])
        # eval_dataset = torch.utils.data.Subset(total_dataset.getEvalData(), [i for i in range(5)])
        # test_dataset = torch.utils.data.Subset(total_dataset.getTestData(), [i for i in range(5)])
        train_dataset = total_dataset.getTrainData()
        test_dataset = total_dataset.getTestData()

        # Prompt

        temperature = args.temperature
        dialog_max_lenght = 1024
        use_temperature = True if temperature > 0 else False
        k = args.k

        model.resize_token_embeddings(len(tokenizer))

        examples = get_examples(train_dataset, k)

        base_prompt = gen_examples_template(examples)

        tokens = tokenizer.tokenize(str(base_prompt))
        token_count = len(tokens)
        length_max = token_count + dialog_max_lenght

        summaries = []

        for dialog, summary_gold in test_dataset:
            prompt = base_prompt + dialog + "SUMMARY:\n"
            inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
            generate_ids = model.generate(
                **inputs,
                do_sample=use_temperature,
                temperature=temperature,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=length_max,
            )
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
                0
            ]
            # output_summary = output.replace(prompt, "").strip().strip("\n")
            output_summary = output.split("SUMMARY:")[-1].strip().strip("\n")
            summaries.append((output_summary, summary_gold))
            # print(output_summary)
        experiment_name = f"d_{args.dataset_name}_pc_{args.use_paracomet}_t_{temperature}_k_{k}"
        save_to_json(summaries, f"summaries_{experiment_name}")
        metrics = compute_metrics(summaries)
        save_to_json(metrics, f"metrics_{experiment_name}")
    finally:
        wandb.finish()


def test():
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
    trainds = total_dataset.getTrainData()
    elements = [(dialog, summary) for dialog, summary in trainds]
    tokens = map(lambda e: tokenizer.tokenize(e[0] + e[1]), elements)
    lens = list(map(lambda t: len(t), tokens))
    avg = sum(lens) / len(lens)
    print(avg)


if __name__ == "__main__":
    a = [("aaaa", "bbbbb"), ("cccc", "ddddd")]
    compute_metrics(a)
    # main()
