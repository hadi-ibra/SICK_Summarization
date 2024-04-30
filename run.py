import time
from src.config.args import get_parser
from argparse import Namespace
import nltk
import numpy as np
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
from dataclasses import replace

from src.data.dataset import SamsumDataset_total, DialogsumDataset_total

from src.config.enums import (
    DatasetOptions,
    ExperimentPhase,
    FrameworkOption,
    ModelCheckpointOptions,
)
from src.experiments.sick import SickExperiment
from src.logging.local_logger import LocalLoggerDecorator
from src.logging.logger import DummyLogger, Logger
from src.logging.wandb_logger import WandbLoggerDecorator
from src.models.bart import BartForConditionalGeneration_DualDecoder
from src.experiments.few_shot import FewShotLearning

nltk.download("punkt")


def get_args() -> Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args


def is_cuda_available() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("Current cuda device:", torch.cuda.current_device())
        print("Count of using GPUs:", torch.cuda.device_count())
        print(torch.cuda.get_device_name())
    return device


def get_datasets(args, tokenizer):
    use_extra_supervision = (args.framework == FrameworkOption.BASIC_SICK_PLUS_PLUS) or (
        args.framework == FrameworkOption.IDIOM_SICK_PLUS_PLUS
    )
    if args.dataset_name == DatasetOptions.SAMSUM:
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
            extra_supervision=use_extra_supervision,
            idiom=args.idiom,
            is_llm=args.is_llm,
        )
        train_dataset = total_dataset.getTrainData()
        eval_dataset = total_dataset.getEvalData()
        test_dataset = total_dataset.getTestData()
    elif args.dataset_name == DatasetOptions.DIALOGSUM:
        total_dataset = DialogsumDataset_total(
            args.encoder_max_len,
            args.decoder_max_len,
            tokenizer,
            extra_context=True,
            paracomet=args.use_paracomet,
            relation=args.relation,
            supervision_relation=args.supervision_relation,
            sentence_transformer=args.use_sentence_transformer,
            extra_supervision=use_extra_supervision,
            roberta=args.use_roberta,
            idiom=args.idiom,
            is_llm=args.is_llm,
        )
        train_dataset = total_dataset.getTrainData()
        eval_dataset = total_dataset.getEvalData()
        test_dataset = total_dataset.getTestData()
    elif args.dataset_name == DatasetOptions.SAMSUM_DEBUG:
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
            extra_supervision=use_extra_supervision,
            idiom=args.idiom,
            is_llm=args.is_llm,
        )
        train_dataset = torch.utils.data.Subset(total_dataset.getTrainData(), [i for i in range(10)])
        eval_dataset = torch.utils.data.Subset(total_dataset.getEvalData(), [i for i in range(5)])
        test_dataset = torch.utils.data.Subset(total_dataset.getTestData(), [i for i in range(5)])
    elif args.dataset_name == DatasetOptions.DIALOGSUM_DEBUG:
        total_dataset = DialogsumDataset_total(
            args.encoder_max_len,
            args.decoder_max_len,
            tokenizer,
            extra_context=True,
            paracomet=args.use_paracomet,
            relation=args.relation,
            supervision_relation=args.supervision_relation,
            sentence_transformer=args.use_sentence_transformer,
            extra_supervision=use_extra_supervision,
            roberta=args.use_roberta,
            idiom=args.idiom,
            is_llm=args.is_llm,
        )
        train_dataset = torch.utils.data.Subset(total_dataset.getTrainData(), [i for i in range(10)])
        eval_dataset = torch.utils.data.Subset(total_dataset.getEvalData(), [i for i in range(5)])
        test_dataset = torch.utils.data.Subset(total_dataset.getTestData(), [i for i in range(5)])

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size is : {len(eval_dataset)}")
    print(f"Test dataset size is : {len(test_dataset)}")

    return total_dataset, train_dataset, eval_dataset, test_dataset


def get_model(args: Namespace, tokenizer, device):
    print(f"Initializing Model")
    if args.load_checkpoint:
        return load_checkpoint(args)
    else:
        return load_pretrained_model(args, tokenizer, device)


def load_checkpoint(args: Namespace):
    if args.framework == FrameworkOption.BASIC_SICK:
        return BartForConditionalGeneration.from_pretrained(args.model_checkpoint)
    elif args.framework == FrameworkOption.BASIC_SICK_PLUS_PLUS:
        return BartForConditionalGeneration_DualDecoder.from_pretrained(args.model_checkpoint)
    else:
        raise NotImplementedError(f"The frameworkOption {args.framework} does not provide checkpoint option")


def load_pretrained_model(args: Namespace, tokenizer, device):
    if args.framework == FrameworkOption.BASIC_SICK:
        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map=device)
        pretrained_model.resize_token_embeddings(len(tokenizer))
        pretrained_model.gradient_checkpointing_enable()
        return pretrained_model
    elif args.framework == FrameworkOption.BASIC_SICK_PLUS_PLUS:
        pretrained_model = BartForConditionalGeneration_DualDecoder.from_pretrained(args.model_name)
        pretrained_model.resize_token_embeddings(len(tokenizer))
        pretrained_model.gradient_checkpointing_enable()
        pretrained_model = pretrained_model.to(device)
        return pretrained_model
    elif args.framework == FrameworkOption.FEW_SHOT:
        # torch.float16 is usable only on supported GPU (basically everying that load llama)
        # it will not run on cpu, but we will not try it anyway
        return AutoModelForCausalLM.from_pretrained(
            args.model_name,
            use_auth_token=args.hugging_face_token,
            torch_dtype=torch.float16,
            device_map=device,
        )
    elif args.framework == FrameworkOption.IDIOM_SICK:
        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map=device)
        pretrained_model.resize_token_embeddings(len(tokenizer))
        pretrained_model.gradient_checkpointing_enable()
        return pretrained_model
    elif args.framework == FrameworkOption.IDIOM_SICK_PLUS_PLUS:
        pretrained_model = BartForConditionalGeneration_DualDecoder.from_pretrained(args.model_name)
        pretrained_model.resize_token_embeddings(len(tokenizer))
        pretrained_model.gradient_checkpointing_enable()
        pretrained_model = pretrained_model.to(device)
        return pretrained_model
    else:
        raise NotImplementedError(f"The model {args.model_name} is not implemented")


def get_tokenizer(args: Namespace):
    print(f"Initializing Tokenizer")
    if args.load_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=args.hugging_face_token)
    if args.framework == FrameworkOption.FEW_SHOT:
        # This line is for debug we have no idea why we should use it
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    special_tokens_dict = {"additional_special_tokens": ["<I>", "</I>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def get_logger(args: Namespace) -> Logger:
    logger = DummyLogger(args)

    if not args.not_use_local_logging:
        logger = LocalLoggerDecorator(logger)
    if not args.not_use_wandb:
        logger = WandbLoggerDecorator(logger)

    return logger


def get_config(args: Namespace) -> Seq2SeqTrainingArguments:
    """Returns the base configuration for SICK-related training."""
    is_fp16_available = torch.cuda.is_available()
    return Seq2SeqTrainingArguments(
        output_dir=args.finetune_weight_path,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        do_predict=False,
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        learning_rate=args.init_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_eps,
        num_train_epochs=args.epoch,
        max_grad_norm=0.1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        lr_scheduler_type="polynomial",
        warmup_steps=args.warm_up,
        save_total_limit=1,
        fp16=is_fp16_available,
        seed=args.seed,
        greater_is_better=True,
        report_to=["wandb"],
    )


def get_finetune_args(args: Namespace) -> Seq2SeqTrainingArguments:
    if args.framework == FrameworkOption.BASIC_SICK or args.framework == FrameworkOption.IDIOM_SICK:
        config = get_config(args)
        options = {
            "load_best_model_at_end": True,
            "predict_with_generate": True,
            "prediction_loss_only": False,
            "generation_max_length": 100,
            "generation_num_beams": 5,
            "metric_for_best_model": "eval_rouge1",
            "do_eval": True,
            "do_predict": True,
            "evaluation_strategy": "epoch",
        }
        config = replace(config, **options)
        return config
    elif (
        args.framework == FrameworkOption.BASIC_SICK_PLUS_PLUS or args.framework == FrameworkOption.IDIOM_SICK_PLUS_PLUS
    ):
        return get_config(args)
    else:
        raise NotImplementedError(f"The finetune args for framework {args.framework} is not implemented")


def main():
    start_time = time.time()
    args = get_args()
    if args.framework == FrameworkOption.IDIOM_SICK or args.framework == FrameworkOption.IDIOM_SICK_PLUS_PLUS:
        args.idiom = True
    device = is_cuda_available()
    gen = np.random.default_rng(args.seed)
    tokenizer = get_tokenizer(args)
    total_dataset, train_dataset, eval_dataset, test_dataset = get_datasets(args, tokenizer)
    model = get_model(args, tokenizer, device)

    test_kwargs = {}

    try:
        logger = get_logger(args)

        is_test_ds_dialog_sum = (args.dataset_name == DatasetOptions.DIALOGSUM) or (
            args.dataset_name == DatasetOptions.DIALOGSUM_DEBUG
        )

        # TODO: discuss the term framework, it's not the best in this context
        if args.framework == FrameworkOption.FEW_SHOT:
            experiment = FewShotLearning(
                model=model,
                train_ds=train_dataset,
                eval_ds=eval_dataset,
                test_ds=test_dataset,
                tokenizer=tokenizer,
                temperature=args.temperature,
                k=args.k,
                gen=gen,
                device=device,
                logger=logger,
            )
        elif args.framework == FrameworkOption.BASIC_SICK or args.framework == FrameworkOption.IDIOM_SICK:
            finetune_args = get_finetune_args(args)
            test_kwargs["num_beams"] = args.num_beams
            experiment = SickExperiment(
                model=model,
                finetune_args=finetune_args,
                freeze_encoder=args.freeze_encoder,
                train_ds=train_dataset,
                eval_ds=eval_dataset,
                test_ds=test_dataset,
                tokenizer=tokenizer,
                gen=gen,
                device=device,
                logger=logger,
                is_plus_version=False,
                is_test_ds_dialog_sum=is_test_ds_dialog_sum,
            )
        elif (
            args.framework == FrameworkOption.BASIC_SICK_PLUS_PLUS
            or args.framework == FrameworkOption.IDIOM_SICK_PLUS_PLUS
        ):
            finetune_args = get_finetune_args(args)
            test_kwargs["num_beams"] = args.num_beams
            experiment = SickExperiment(
                model=model,
                finetune_args=finetune_args,
                freeze_encoder=args.freeze_encoder,
                train_ds=train_dataset,
                eval_ds=eval_dataset,
                test_ds=test_dataset,
                tokenizer=tokenizer,
                gen=gen,
                device=device,
                logger=logger,
                is_plus_version=True,
                is_test_ds_dialog_sum=is_test_ds_dialog_sum,
            )
        else:
            raise NotImplementedError("The framework selected is not implemented")

        if args.phase == ExperimentPhase.ALL:
            experiment.train()
            saving_object = experiment.save()
            if saving_object is not None:
                logger.save(saving_object)
            experiment.test(**test_kwargs)
        elif args.phase == ExperimentPhase.TRAIN:
            experiment.train()
            saving_object = experiment.save()
            logger.save(saving_object)
        elif args.phase == ExperimentPhase.TEST:
            experiment.test(**test_kwargs)
        else:
            raise NotImplementedError("The phase chosen is not implemented")

    finally:
        logger.finish()
        torch.cuda.empty_cache()
        end_time = time.time()
        print(f"Elapsed time: {round(end_time - start_time, 2)}")


if __name__ == "__main__":
    main()
