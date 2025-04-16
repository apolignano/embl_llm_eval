import argparse
import os
import json
import logging
from tqdm import tqdm
import warnings

from task import *
from scoring import *
from preprocess import *
from llm import ask_llm
from openai import Client

warnings.filterwarnings("ignore")

# Load clients
ollama_client = Client(base_url="http://localhost:11434")
openai_client = Client()
clients = {"ollama": ollama_client, "openai": openai_client}
logging.info("Clients loaded successfully.")

parser = argparse.ArgumentParser(description="A simple command-line script.")
parser.add_argument(
    "--task-folder",
    type=str,
    required=True,
    help="Folder with task .yml configuration and .csv dataset",
)
parser.add_argument(
    "--results-folder",
    type=str,
    required=True,
    help="Folder where the benchmark results will be saved",
)

args = parser.parse_args()

config_path, dataset_path = validate_task_folder(args.task_folder)
task_path = args.task_folder
task_config = validate_config(config_path)

task_name = task_config["task_name"]
models_list = task_config["models_list"]
prompt_template = task_config["prompt"]

preprocess = task_config["preprocess"]
preprocess_params = preprocess["params"]
preprocess_func_name = preprocess["name"]
preprocess_func = import_task_func(task_path, preprocess_func_name, "preprocess")

subinput = task_config["subinput"] if task_config["subinput"] else None
if (subinput):
    subinput_splitter = import_task_func(task_path, subinput["splitter"], "preprocess")
    subinput_aggregator = import_task_func(task_path, subinput["aggregator"], "preprocess")

input_fetcher = task_config["input_fetcher"] if task_config["subinput"] else None
input_fetcher_func = import_task_func(task_path, input_fetcher, "preprocess")

scoring = task_config["scoring"]
aggregation = task_config["aggregation"]

# aggregation = task_config["aggregation"]
logging.info("Config loaded successfully.")

def main():

    df = validate_dataset(dataset_path, task_config["dataset_separator"])
    inputs = df["input"].to_list()
    targets = df["target"].to_list()
    param_columns = [v for v in df.drop(columns=["input", "target"]).T.to_dict().values()] 

    # Get type of models
    model_types = json.load(open("model_type.json"))

    # Generate answers from list of models
    model_to_answers = {k: {} for k in models_list}

    for model in models_list:

        print(f"Evaluating {model}...")

        answers = []
        unfiltered_answers = []
        for i in tqdm(np.arange(0, len(inputs), 1)):
            input_text = inputs[i]
            params = param_columns[i]

            if (input_fetcher):
                input_text = input_fetcher_func(input_text)

            client_type = model_types[model]

            if (subinput):
                splitted_input = subinput_splitter(input_text)
                unfiltered_subinput_answers = []
                subinput_answers = []
                for sub_input in tqdm(splitted_input):
                    response = ask_llm(clients, client_type, model, prompt_template, sub_input, params)
                    unfiltered_subinput_answers.append(response)
                    subinput_answers.append(preprocess_func(response, **preprocess_params))
                
                # Aggregate subinput answers
                unfiltered_answers.append(unfiltered_subinput_answers)
                answers.append(subinput_aggregator(subinput_answers))

            else: 
                response = ask_llm(clients, client_type, model, prompt_template, input_text, params)
                unfiltered_answers.append(response)
                answers.append(preprocess_func(response, **preprocess_params))

        model_to_answers[model]["answers"] = answers
        model_to_answers[model]["unfiltered_answers"] = unfiltered_answers


    for model in models_list:
        model_to_answers[model]["scores"] = {}
        model_to_answers[model]["aggregates"] = {}
        zipped_answers_targets = list(zip(model_to_answers[model]["answers"], targets))
        
        # Compute per-sample scores
        for scoring_func_name in scoring:
            scoring_func = import_task_func(task_path, scoring_func_name, "scoring")
            model_to_answers[model]["scores"][scoring_func_name] = [scoring_func(answer, target) for answer, target in zipped_answers_targets]

            # Compute aggregation scores
            for agg_func_name in aggregation:
                agg_func = import_task_func(task_path, agg_func_name, "scoring")
                model_to_answers[model]["aggregates"][f"{agg_func_name}_{scoring_func_name}"] = agg_func(model_to_answers[model]["scores"][scoring_func_name])


    # Create results folder if it doesn't exist and save results
    create_results_folder(args.results_folder)
    json.dump(
        {"targets": targets, "model_results": model_to_answers},
        open(os.path.join(args.results_folder, "results.json"), "w"),
        indent=2,
    )

if __name__ == "__main__":
    main()