# Mind the Gap! Injecting Commonsense Knowledge for Abstractive Dialogue Summarization
In this project we expand on the paper https://arxiv.org/abs/2209.00930, starting from reproducing the results obtained in the original research and then performing three extensions to evaluate correlated strategies. All the details about the work done and the results obtained are visible in the apposite report.

Original repository : https://github.com/SeungoneKim/SICK_Summarization

Overview of base method, SICK (Summarizing with Injected Commonsense Knowledge).
<p align="center">
  <img src="./SICK_overview.png" width="100%" height="80%">
</p>

## Setting
The project can be run both in a local environment (if at least a GPU is present) or on a google colab. You can follow the steps presented in the sections [Local environment](README.md#local-environment) and [Experiments run](README.md#experiments-run)

### Local environment
1. Clone the project repository
   ``` git clone https://github.com/hadi-ibra/SICK_Summarization```
2. Create an enviroment using conda or virtualenv (instructions will consider only conda)
   ```
   conda create -n sick python=3.10
   conda activate sick
   pip install -r requirements.txt
   pip install -U spacy
   pip install accelerate -U
   python -m spacy download en_core_web_sm
   ```
3. Add the needed datasets
   - SamSum: already provided with the library Hugging Face
   - DialogSum: download it from [here](https://drive.google.com/drive/folders/1arrpEPCn31ZQ7bKlSC3hJZxocQmeZFlV?usp=sharing) and copy it under ```data/DialogSum_Data```
   - SamSum and DialogSum commonsense: download it from [here](https://drive.google.com/drive/folders/1CK8cfOdw5gFmhrfjuEu6vAWybPeUR2G2?usp=sharing) and copy it under ```data/COMET_data```
4. Go to the run section

### Google colab
1. Clone the project repository
   ``` git clone https://github.com/hadi-ibra/SICK_Summarization```
2. Install the required libraries
   ```
   pip install -r requirements.txt
   pip install -U spacy
   pip install accelerate -U
   python -m spacy download en_core_web_sm
   ```
3. Add the needed datasets
   - SamSum: already provided with the library Hugging Face
   - DialogSum: download it from [here](https://drive.google.com/drive/folders/1arrpEPCn31ZQ7bKlSC3hJZxocQmeZFlV?usp=sharing) and copy it under ```data/DialogSum_Data```
   - SamSum and DialogSum commonsense: download it from [here](https://drive.google.com/drive/folders/1CK8cfOdw5gFmhrfjuEu6vAWybPeUR2G2?usp=sharing) and copy it under ```data/COMET_data```
4. Go to the run section

## Demo
Inside the project is present a ```demo.py``` file to run a pipeline with 4 different model (SICK, SICK+Idiom, Few-Shot (Llama), Few-Shot+Idiom (Llama)) on a dialog extracted from SamSum.
A helper jupyter notebook is also present, called ```demo_runner.ipynb```. It contains all the required preliminary operation described in the "Setting" section with just a few path to modify.

## Experiments run
Use the following command to run all the project experiments. Every experiment category present a subset of possible parameters.
```
python3 run.py <PARAMS>
```
Before running the experiment we need to performe the wandb login if we want to be used as logger. Use ```wandb login <TOKEN>``` adding your own wandb token. In case we don't want to use it, just add ```--not_use_wandb``` in your arguments.

To run variation of the presented categories with the idiom dataset change the value of ```---framework``` with the one reported:
 - "basic_sick" => "idiom_sick"
 - "basic_sick_plus_plus" => "idiom_sick_plus_plus"
 - "few_shot_learning" => "idiom_few_shot"
and add ```--idiom True``` in your arguments.

### SICK example run
```
wandb disabled
python3 run.py --hugging_face_token <ADD_YOUR_TOKEN> --project sick_samsum --framework "basic_sick" --exp_name sick_samsum_5e_paracomet --seed 516 --phase "all" --dataset_name "samsum" --model_name "facebook/bart-large-xsum" --finetune_weight_path "./weights_sick_samsum" --best_finetune_weight_path "./weights_sick_samsum_best" --epoch 5 --use_paracomet True --relation "xIntent" --use_sentence_transformer True --not_use_wandb
```

For SICK some configuration are:
 - hugging_face_token <HUGGING FACE KEY>: Token to access Hugging Face and download the needed components.
 - project <PROJECT NAME>: Name of the project to use when loggin with wandb. If wandb is not used the argument is ignored.
 - framework "basic_sick": Type of experiments to performe (see src/config/enums.py/FrameworkOption).
 - exp_name <ADD_EXP_NAME>: Name of the experiment. It will used for the run in wandb and as part of the name for the local logger (after data).
 - seed 516: Seed to use for the run (all the experiments reported in the report are performed using 516 as value).
 - phase <PHASE>: Phases to performe during the experiment (see enums.py/ExperimentPhase - for training + test use "all").
 - dataset_name <DATASET_NAME>: Name of the dataset to use (see src/config/enums.py/DatasetOptions).
 - model_name <MODEL_NAME>: Model to use for the run (see src/config/enums.py/ModelCheckpointOptions).
 - finetune_weight_path <FOLDER_NAME>: Path/Folder where is going to be stored weights while training.
 - best_finetune_weight_path <FOLDER_NAME>: Path/Folder where is going to be stored weights after training finished.
 - epoch: Number of epochs used during training if performed.
 - use_paracomet: If you set to true, it will use paracomet, and if not, it will use comet by default.
 - relation: If you would only like to use one of the 5 possible relations, you could specify it with this argument.
 - use_sentence_transformer: If you would like to use the commonsense selected with sentence_transformer, you should use this argument.

### SICK++ example run
```
wandb disabled
python3 run.py --hugging_face_token <ADD_YOUR_TOKEN> --project base_sick --framework "basic_sick_plus_plus" --exp_name sickplus_samsum_5e --seed 516 --phase "all" --dataset_name "samsum_debug" --model_name "facebook/bart-large-xsum" --finetune_weight_path "./weights_sickplus_samsum" --best_finetune_weight_path "./weights_sickplus_samsum_best" --epoch 5 --use_paracomet True --relation "xIntent" --use_sentence_transformer True --not_use_wand --supervision_relation "xIntent"
```

For SICK++ some configuration are:
 - hugging_face_token <HUGGING FACE KEY>: Token to access Hugging Face and download the needed components.
 - project <PROJECT NAME>: Name of the project to use when loggin with wandb. If wandb is not used the argument is ignored.
 - framework "basic_sick_plus_plus": Type of experiments to performe (see src/config/enums.py/FrameworkOption).
 - exp_name <ADD_EXP_NAME>: Name of the experiment. It will used for the run in wandb and as part of the name for the local logger (after data).
 - seed 516: Seed to use for the run (all the experiments reported in the report are performed using 516 as value).
 - phase <PHASE>: Phases to performe during the experiment (see enums.py/ExperimentPhase - for training + test use "all").
 - dataset_name <DATASET_NAME>: Name of the dataset to use (see src/config/enums.py/DatasetOptions).
 - model_name <MODEL_NAME>: Model to use for the run (see src/config/enums.py/ModelCheckpointOptions).
 - finetune_weight_path <FOLDER_NAME>: Path/Folder where is going to be stored weights while training.
 - best_finetune_weight_path <FOLDER_NAME>: Path/Folder where is going to be stored weights after training finished.
 - epoch: Number of epochs used during training if performed.
 - use_paracomet: If you set to true, it will use paracomet, and if not, it will use comet by default.
 - relation: If you would only like to use one of the 5 possible relations, you could specify it with this argument.
 - use_sentence_transformer: If you would like to use the commonsense selected with sentence_transformer, you should use this argument.
 - supervision_relation : If you would only like to use one of the 5 possible supervision relations, you could specify it with this argument.

### FewShot example run
```
wandb disabled
python3 run.py --hugging_face_token <ADD_YOUR_TOKEN> --project "samsum_t0_k4" --framework "few_shot_learning" --exp_name "samsum_t0_k4" --seed 516 --phase "all" --dataset_name "samsum" --model_name "meta-llama/Llama-2-7b-chat-hf" --epoch 1 --use_paracomet True --relation "xIntent" --use_sentence_transformer True --temperature 0 --k 4 --is_llm Tru --not_use_wandb
```
For FewShot some configuration are:
 - hugging_face_token <HUGGING FACE KEY>: Token to access Hugging Face and download the needed components.
 - project <PROJECT NAME>: Name of the project to use when loggin with wandb. If wandb is not used the argument is ignored.
 - framework "few_shot_learning": Type of experiments to performe (see src/config/enums.py/FrameworkOption).
 - exp_name <ADD_EXP_NAME>: Name of the experiment. It will used for the run in wandb and as part of the name for the local logger (after data).
 - seed 516: Seed to use for the run (all the experiments reported in the report are performed using 516 as value).
 - phase <PHASE>: Phases to performe during the experiment (see enums.py/ExperimentPhase - for training + test use "all").
 - dataset_name <DATASET_NAME>: Name of the dataset to use (see src/config/enums.py/DatasetOptions).
 - model_name <MODEL_NAME>: Model to use for the run (see src/config/enums.py/ModelCheckpointOptions).
 - temperature <VALUE>: Value of temperature to use in the model.
 - k <VALUE>: Number of examples to add in the prompt.
 - is_llm True: Indicate the use of an LLM as a model, required to get the right format of data from the dataset.

### Evaluation
If you want to perform the evaluation of a previously trained model, run the command ```python3 run.py <PARAMS>``` with the following arguments:
 - hugging_face_token <HUGGING FACE KEY>: Token to access Hugging Face and download the needed components.
 - project <PROJECT NAME>: Name of the project to use when loggin with wandb. If wandb is not used the argument is ignored.
 - framework <FRAMEWORK_NAME>: Type of experiments to performe (see src/config/enums.py/FrameworkOption).
 - exp_name <ADD_EXP_NAME>: Name of the experiment. It will used for the run in wandb and as part of the name for the local logger (after data).
 - seed 516: Seed to use for the run (all the experiments reported in the report are performed using 516 as value).
 - phase "test": Phases to performe during the experiment (see enums.py/ExperimentPhase - for training + test use "all").
 - dataset_name <DATASET_NAME>: Name of the dataset to use (see src/config/enums.py/DatasetOptions).
 - load_checkpoint True: Indicate to the problem to load the model from a checkpoint
 - model_checkpoint <VALUE>: Path/Folder where the model checkpoint is stored
