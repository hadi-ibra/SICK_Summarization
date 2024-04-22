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
python -m spacy download en_core_web_sm

# Step 3: 
# login inside wandb with api key if needed for logging (be careful to add/adjust program params accordingly with this) 
# echo "[SCRIPT]: Login in wandb"
# wandb login <API_KEY>
# echo "[SCRIPT]: Login done"

# Step 4: 
# the main entry point for the repo is in run.py. Configure the run with the provided argument parser.
# Note 1: for enums use the value of them (e.g. for framework use "few_shot_learning" and not "FEW_SHOT").
# Note 2: if needed run python3 src/run.py --help to recive guide on the parameters available
echo "[SCRIPT]: Starting the run"
PARAMS=(
    --hugging_face_token <HUGGING FACE KEY>
    # project name in wandb
    --project <PROJECT NAME> # look at project available on wandb. If wandb is not used put whatever you want
    --framework <FRAMEWORK> # (see src/config/enums.py/FrameworkOption)
    --exp_name <ADD_EXP_NAME> # it will used for the run in wandb and as part of the name for the local logger (after data)
    --seed 516
    --phase test
    --dataset_name <DATASET_NAME> # (see src/config/enums.py/DatasetOptions)
    --load_checkpoint True
    --model_checkpoint <PATH> # Path/Folder where the model is saved
    --use_paracomet True
    --relation "xIntent"
    --use_sentence_transformer True
)

python3 run.py "${PARAMS[@]}"

echo "[SCRIPT]: Run ended"