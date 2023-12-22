import enum

class ModelCheckpointOptions(str, enum.Enum):
    BART_LARGE = "facebook/bart-large", 
    BART_LARGE_XSUM = "facebook/bart-large-xsum",
    PEGASUS_LARGE = "google/pegasus-large",
    PEGASUS_XSUM = "google/pegasus-xsum",
    T5_LARGE_LM = "google/t5-large-lm-adapt", 
    T5_V1_LARGE = "google/t5-v1_1-large"
    
class DatasetOptions(str, enum.Enum):
    SAMSUM = "samsum",
    DIALOGSUM = "dialogsum",