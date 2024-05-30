# Disable all library warning for the demo
import warnings

warnings.filterwarnings("ignore")

from argparse import Namespace
import time
import numpy as np
import torch
import os

from run import get_finetune_args, get_model, get_tokenizer, is_cuda_available, load_checkpoint, load_pretrained_model

from src.config.args import get_parser
from src.config.enums import FrameworkOption
from src.data.dataset import SamsumDataset_total
from src.experiments.few_shot import FewShotLearning
from src.experiments.sick import SickExperiment
from src.logging.demo_logger import DemoLogger
from src.logging.local_logger import LocalLoggerDecorator
from src.logging.logger import DummyLogger, Logger
from src.logging.wandb_logger import WandbLoggerDecorator


def get_demo_dataset(tokenizer, config):
    use_extra_supervision = (args.framework == FrameworkOption.BASIC_SICK_PLUS_PLUS) or (
        args.framework == FrameworkOption.IDIOM_SICK_PLUS_PLUS
    )
    total_dataset = SamsumDataset_total(
        config.encoder_max_len,
        config.decoder_max_len,
        tokenizer,
        extra_context=config.use_commonsense,
        paracomet=config.use_paracomet,
        relation=config.relation,
        supervision_relation=config.supervision_relation,
        roberta=config.use_roberta,
        sentence_transformer=config.use_sentence_transformer,
        extra_supervision=use_extra_supervision,
        idiom=config.idiom,
        is_llm=config.is_llm,
    )

    # TODO: change indexes for the sub-dataset
    train_dataset = torch.utils.data.Subset(total_dataset.getTrainData(), [i for i in range(10)])
    eval_dataset = torch.utils.data.Subset(total_dataset.getEvalData(), [i for i in range(5)])
    test_dataset = torch.utils.data.Subset(total_dataset.getTestData(), [18])
    # train_dataset = total_dataset.getTrainData()
    # eval_dataset = total_dataset.getEvalData()
    # test_dataset = total_dataset.getTestData()
    return train_dataset, eval_dataset, test_dataset


def get_demo_model(config, is_checkpoint, tokenizer, device):
    print(f"Initializing Model")
    if is_checkpoint:
        return load_checkpoint(config)
    else:
        return load_pretrained_model(config, tokenizer, device)


def get_logger(args: Namespace) -> Logger:
    logger = DummyLogger(args)

    if not args.not_use_local_logging:
        logger = LocalLoggerDecorator(logger)
    if not args.not_use_wandb:
        logger = WandbLoggerDecorator(logger)

    logger = DemoLogger(logger)

    return logger


def main(config: Namespace):
    start_time = time.time()

    # Configure pipeline

    # General things
    device = is_cuda_available()
    gen = np.random.default_rng(config.seed)

    tokenizer = get_tokenizer(config)

    train_dataset, eval_dataset, test_dataset = get_demo_dataset(tokenizer, config)

    # model = get_demo_model(config, config.is_checkpoint, tokenizer, device)
    model = get_model(config, tokenizer, device)

    test_kwargs = {}

    try:

        logger = get_logger(config)

        if config.framework == FrameworkOption.FEW_SHOT or config.framework == FrameworkOption.FEW_SHOT_IDIOM:
            experiment = FewShotLearning(
                model=model,
                train_ds=train_dataset,
                eval_ds=eval_dataset,
                test_ds=test_dataset,
                tokenizer=tokenizer,
                temperature=config.temperature,
                k=config.k,
                gen=gen,
                device=device,
                logger=logger,
                is_test_ds_dialog_sum=False,
            )
        elif config.framework == FrameworkOption.BASIC_SICK or config.framework == FrameworkOption.IDIOM_SICK:
            finetune_args = get_finetune_args(config)
            test_kwargs["num_beams"] = config.num_beams
            experiment = SickExperiment(
                model=model,
                finetune_args=finetune_args,
                freeze_encoder=config.freeze_encoder,
                train_ds=train_dataset,
                eval_ds=eval_dataset,
                test_ds=test_dataset,
                tokenizer=tokenizer,
                gen=gen,
                device=device,
                logger=logger,
                is_plus_version=False,
                is_test_ds_dialog_sum=False,
            )
        elif (
            config.framework == FrameworkOption.BASIC_SICK_PLUS_PLUS
            or config.framework == FrameworkOption.IDIOM_SICK_PLUS_PLUS
        ):
            finetune_args = get_finetune_args(config)
            test_kwargs["num_beams"] = config.num_beams
            experiment = SickExperiment(
                model=model,
                finetune_args=finetune_args,
                freeze_encoder=config.freeze_encoder,
                train_ds=train_dataset,
                eval_ds=eval_dataset,
                test_ds=test_dataset,
                tokenizer=tokenizer,
                gen=gen,
                device=device,
                logger=logger,
                is_plus_version=True,
                is_test_ds_dialog_sum=False,
            )

        experiment.test(**test_kwargs)

    finally:
        results = logger.get_saved_results()  # type: ignore
        logger.finish()
        end_time = time.time()
        torch.cuda.empty_cache()
        print(f"Elapsed time: {round(end_time - start_time, 2)}")
        return results


if __name__ == "__main__":
    try:
        hugging_face_token = os.environ["HUGGING_FACE_TOKEN"]
    except KeyError:
        print("ERROR: the HUGGING_FACE_TOKEN is not provided as environment variable")
        exit(1)
    parser = get_parser()
    configs = {
        "Normal Sick": [
            f"--hugging_face_token={hugging_face_token}",
            "--project=demo",
            "--framework=idiom_sick",
            "--exp_name=idiom_sick_demo",
            "--seed=516",
            "--phase=test",
            "--dataset_name=samsum",
            "--model_name=facebook/bart-large-xsum",
            "--use_paracomet=True",
            "--relation=xIntent",
            "--use_sentence_transformer=True",
            "--idiom=False",
            "--load_checkpoint=True",
            "--model_checkpoint=./sick_best",
            "--not_use_local_logging",
            "--not_use_wandb",
        ],
        "Idiom Sick": [
            f"--hugging_face_token={hugging_face_token}",
            "--project=demo",
            "--framework=idiom_sick",
            "--exp_name=idiom_sick_demo",
            "--seed=516",
            "--phase=test",
            "--dataset_name=samsum",
            "--model_name=facebook/bart-large-xsum",
            "--use_paracomet=True",
            "--relation=xIntent",
            "--use_sentence_transformer=True",
            "--idiom=True",
            "--load_checkpoint=True",
            "--model_checkpoint=./idiom_sick_best",
            "--not_use_local_logging",
            "--not_use_wandb",
        ],
        "Normal FewShot t=0, k=2": [
            f"--hugging_face_token={hugging_face_token}",
            "--project=few_shot_idiom",
            "--framework=idiom_few_shot",
            "--exp_name=idiom_samsum_t0_k2",
            "--seed=516",
            "--phase=all",
            "--dataset_name=samsum",
            "--model_name=meta-llama/Llama-2-7b-chat-hf",
            "--epoch=1",
            "--use_paracomet=True",
            "--relation=xIntent",
            "--use_sentence_transformer=True",
            "--temperature=0",
            "--k=2",
            "--is_llm=True",
            "--idiom=False",
        ],
        "Idiom FewShot t=0, k=2": [
            f"--hugging_face_token={hugging_face_token}",
            "--project=few_shot_idiom",
            "--framework=idiom_few_shot",
            "--exp_name=idiom_samsum_t0_k2",
            "--seed=516",
            "--phase=all",
            "--dataset_name=samsum",
            "--model_name=meta-llama/Llama-2-7b-chat-hf",
            "--epoch=1",
            "--use_paracomet=True",
            "--relation=xIntent",
            "--use_sentence_transformer=True",
            "--temperature=0",
            "--k=2",
            "--is_llm=True",
            "--idiom=True",
        ],
    }
    pipelines_results = []
    for key, config in configs.items():
        print(f"Running config: {key}")
        args = parser.parse_args(config)
        pipeline_result = main(args)
        pipelines_results.append(pipeline_result)

    for k, result in zip(configs.keys(), pipelines_results):
        print(k)
        print(result)
