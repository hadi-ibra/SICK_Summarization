import enum


class ModelCheckpointOptions(str, enum.Enum):
    BART_LARGE = "facebook/bart-large"
    BART_LARGE_XSUM = "facebook/bart-large-xsum"
    PEGASUS_LARGE = "google/pegasus-large"
    PEGASUS_XSUM = "google/pegasus-xsum"
    T5_LARGE_LM = "google/t5-large-lm-adapt"
    T5_V1_LARGE = "google/t5-v1_1-large"
    LLAMA = "meta-llama/Llama-2-7b-chat-hf"

    def __str__(self) -> str:
        return self.value


class DatasetOptions(str, enum.Enum):
    SAMSUM = "samsum"
    DIALOGSUM = "dialogsum"
    DEBUG = "samsum_debug"

    def __str__(self) -> str:
        return self.value


class FrameworkOption(str, enum.Enum):
    FEW_SHOT = "few_shot_learning"
    BASIC_SICK = "basic_sick"
    BASIC_SICK_PLUS_PLUS = "basic_sick_plus_plus"
    IDIOM_SICK = "idiom_sick"
    IDIOM_SICK_PLUS_PLUS = "idiom_sick_plus_plus"

    def __str__(self) -> str:
        return self.value


class ExperimentPhase(str, enum.Enum):
    ALL = "all"
    TRAIN = "train"
    TEST = "test"

    def __str__(self) -> str:
        return self.value
