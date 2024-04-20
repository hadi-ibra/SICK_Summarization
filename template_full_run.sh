# Launch script template for experiment runs. Uncomment and modify all the needed step/params

# Step 1: 
# clone the repo with the command git clone <REPO> and move in the desired branch (if needed)
# Example: git clone https://github.com/hadi-ibra/SICK_Summarization.git
# Example: git checkout <BRANCH NAME>

# Step2: 
# move inside it and install requirements
cd SICK_Summarization/

# Do not touch the following part
pip install -r requirements.txt
pip install -U spacy
pip install accelerate -U
pip install backports.zoneinfo
python -m spacy download en_core_web_sm


# Step 3: 
# login inside wandb with api key if needed for logging (be careful to add/adjust program params accordingly with this) 
wandb_api_key=$(python3 -c 'import yaml; data = yaml.safe_load(open("src/config/secret.yml")); print(data["WANDB_API_KEY"])')

echo "[SCRIPT]: Login in wandb"
wandb login 
echo "[SCRIPT]: Login done"

# Extract values from the config files
hugging_face_token=$(python3 -c 'import yaml; data = yaml.safe_load(open("src/config/secret.yml")); print(data["HUGGING_FACE_TOKEN"])')
project_name=$(python3 -c 'import yaml; data = yaml.safe_load(open("src/config/params.yml")); print(data["PROJECT_NAME"])')
framework=$(python3 -c 'import yaml; data = yaml.safe_load(open("src/config/params.yml")); print(data["FRAMEWORK"])')
exp_name=$(python3 -c 'import yaml; data = yaml.safe_load(open("src/config/params.yml")); print(data["EXPERIMENT_NAME"])')
phase=$(python3 -c 'import yaml; data = yaml.safe_load(open("src/config/params.yml")); print(data["PHASE"])')
dataset_name=$(python3 -c 'import yaml; data = yaml.safe_load(open("src/config/params.yml")); print(data["DATASET_NAME"])')
model_name=$(python3 -c 'import yaml; data = yaml.safe_load(open("src/config/params.yml")); print(data["MODEL_NAME"])')
folder_name=$(python3 -c 'import yaml; data = yaml.safe_load(open("src/config/params.yml")); print(data["FOLDER_NAME"])')


# Step 4: 
# the main entry point for the repo is in run.py. Configure the run with the provided argument parser.
# Note 1: for enums use the value of them (e.g. for framework use "few_shot_learning" and not "FEW_SHOT").
# Note 2: if needed run python3 src/run.py --help to recive guide on the parameters available
#!/bin/bash


echo "[SCRIPT]: Starting the run"
PARAMS=(
    --hugging_face_token $hugging_face_token
    # project name in wandb
    --project $project_name # look at project available on wandb. If wandb is not used put whatever you want
    --framework $framework # (see src/config/enums.py/FrameworkOption)
    --exp_name $exp_name # it will used for the run in wandb and as part of the name for the local logger (after data)
    --seed 516
    --phase $phase # (see enums.py/ExperimentPhase - for training + test use "all")
    --dataset_name $dataset_name # (see src/config/enums.py/DatasetOptions)
    --model_name $model_name # (see src/config/enums.py/ModelCheckpointOptions)
    --finetune_weight_path $folder_name # Path/Folder where is going to be stored weights while training
    --best_finetune_weight_path $folder_name # Path/Folder where is going to be stored weights after training finished
    --epoch 1
    --use_paracomet True
    --relation "xIntent"
    --use_sentence_transformer True
)

python3 run.py "${PARAMS[@]}"

echo "[SCRIPT]: Run ended"


!bash my_shell_script.sh 