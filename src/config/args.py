import argparse

from src.config.enums import (
    DatasetOptions,
    ExperimentPhase,
    FrameworkOption,
    ModelCheckpointOptions,
)


def get_parser() -> argparse.ArgumentParser:
    # Set Argument Parser
    parser = argparse.ArgumentParser(
        prog="NLP Project",
        description="Program to train and evaluate SICK architecture",
        epilog="Thanks for running me :-)",
    )

    parser.add_argument(
        "--hugging_face_token",
        type=str,
        required=True,
        help="Token to download hugging face models",
    )

    parser.add_argument(
        "--framework",
        type=FrameworkOption,
        choices=list(FrameworkOption),
        help="Type of framework used for experiment (e.g. sick, sick++, few_shot, etc.)",
    )

    parser.add_argument("--project", type=str, required=True, help="Project name")

    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="name of the experiment. It will be used to create the folder containing all the run results and model weights",
    )

    parser.add_argument(
        "--phase",
        type=ExperimentPhase,
        choices=list(ExperimentPhase),
        required=True,
        help="Phase of the experiment",
    )

    parser.add_argument(
        "--not_use_local_logging",
        action="store_true",
        help="disable experiment track with build-in serializer",
    )

    parser.add_argument(
        "--not_use_wandb",
        action="store_true",
        help="disable experiment track with wandb logging",
    )

    parser.add_argument("--seed", type=int, default=516)

    # Training hyperparameters
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=20)
    # parser.add_argument('--display_step',type=int, default=2000)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    # Model hyperparameters
    # parser.add_argument('--model_name',type=ModelCheckpointOptions, choices=list(ModelCheckpointOptions), default='facebook/bart-large-xsum')
    parser.add_argument(
        "--model_name",
        type=ModelCheckpointOptions,
        choices=list(ModelCheckpointOptions),
    )
    parser.add_argument("--freeze_encoder", type=bool, default=False)
    # Optimizer hyperparameters
    parser.add_argument("--init_lr", type=float, default=3e-6)
    parser.add_argument("--warm_up", type=int, default=600)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--decay_epoch", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-12)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    # Tokenizer hyperparameters
    parser.add_argument("--encoder_max_len", type=int, default=1024)
    parser.add_argument("--decoder_max_len", type=int, default=100)
    parser.add_argument("--vocab_size", type=int, default=51201)
    parser.add_argument("--eos_idx", type=int, default=51200)
    parser.add_argument("--tokenizer_name", type=str, default="RobertaTokenizer")
    # Checkpoint directory hyperparameters
    parser.add_argument(
        "--pretrained_weight_path", type=str, default="pretrained_weights"
    )
    parser.add_argument(
        "--finetune_weight_path",
        type=str,
        default="./context_BART_weights_Samsum_5epoch",
    )
    parser.add_argument(
        "--best_finetune_weight_path",
        type=str,
        default="context_final_BART_weights_Samsum_5epoch",
    )
    # Dataset hyperparameters
    parser.add_argument(
        "--dataset_name",
        type=DatasetOptions,
        choices=list(DatasetOptions),
        default="samsum",
    )
    parser.add_argument("--use_paracomet", type=bool, default=False)
    parser.add_argument("--use_roberta", type=bool, default=False)
    parser.add_argument("--use_sentence_transformer", type=bool, default=False)
    parser.add_argument("--dataset_directory", type=str, default="./data")
    parser.add_argument("--relation", type=str, default="xReason")
    parser.add_argument("--supervision_relation", type=str, default="isAfter")

    # Few-shot params
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--is_llm", type=bool, default=False)

    # Inference params
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="./new_weights_comet/final_Trial1_context_comet",
    )
    parser.add_argument("--test_output_file_name", type=str, default="results.json")
    # TODO: Remove --train_configuration (probably not used in the new code, check it)
    parser.add_argument(
        "--train_configuration", type=str, default="full"
    )  # base, context, supervision, full
    parser.add_argument("--num_beams", type=int, default=20)

    return parser
