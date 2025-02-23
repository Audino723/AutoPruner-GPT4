from tqdm import tqdm
import numpy as np
import torch
from src.utils.utils import Logger, AverageMeter, evaluation_metrics, read_config_file
import os
import warnings
import argparse

import numpy as np
from src.gpt_4.dataset import CallGraphDataset
from src.utils.utils import read_config_file
from src.gpt_4.model import GPT4_Model  
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import json


warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PARAMS = {'batch_size': 15, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': 10, 'shuffle': False, 'num_workers': 8}

# main.py
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

def do_test_gpt4(config, dataset):
    """
    Test GPT-4's ability to prune call graphs using the dataset.
    """
    all_outputs = []
    all_labels = []
    results = []
    estimate_cost = 0

    gpt4_model = GPT4_Model()
    START_IDX = 0
    END_IDX =  10

    print("Begin testing...")
    loop = tqdm(enumerate(dataset), leave=False, total=len(dataset))
    for idx, batch in loop:
        if idx < START_IDX:
            continue

        if idx > END_IDX:
            break

        try:
            src = batch['src']
            dst = batch['dst']
            struct = batch['struct']
            label = batch['label'].numpy()

            # Generate prompt and get GPT-4 output
            # gpt4_output, response, prompt = gpt4_model.forward(src, dst, struct)
            gpt4_output, response, prompt = gpt4_model.forward_cot(src, dst, struct)
            pred = gpt4_output

            print(gpt4_output)
            print(response)
            print(label)

            # Store individual outputs and labels
            all_labels.append(label)
            all_outputs.append(pred)

            # Collect detailed result for saving
            results.append({
                "Index": idx,
                "Start": src,
                "Destination": dst,
                "Structure": struct,
                "Label": label,
                "Prediction": pred,
                "Prompt": prompt,
                "Response": response
            })

            # Update progress bar metrics
            if all_labels and all_outputs:
                precision = precision_score(all_labels, all_outputs, zero_division=0)
                recall = recall_score(all_labels, all_outputs, zero_division=0)
                f1 = f1_score(all_labels, all_outputs, zero_division=0)
                loop.set_postfix(pre=precision, rec=recall, f1=f1, cost=gpt4_model.total_cost, total_tokens=gpt4_model.total_usage_tokens)

        except Exception as e:
            print(f"Error occurred at batch {idx}: {e}")
            continue  # Skip the current batch and move to the next

    # Save results to a CSV file
    if results:
        save_dir = config.get("GPT4_CACHE_DIR", "cache")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        output_file = os.path.join(save_dir, f"gpt4_result_{START_IDX}_{END_IDX}.csv")
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    # Compute overall metrics
    overall_precision = precision_score(all_labels, all_outputs, zero_division=0) if all_labels else 0
    overall_recall = recall_score(all_labels, all_outputs, zero_division=0) if all_labels else 0
    overall_f1 = f1_score(all_labels, all_outputs, zero_division=0) if all_labels else 0

    print(f"Total Token Usage : {gpt4_model.total_usage_tokens}")
    print(f"Estimate cost     : {gpt4_model.total_cost:.8f} $")
    print(f"Overall Precision : {overall_precision:.4f}")
    print(f"Overall Recall    : {overall_recall:.4f}")
    print(f"Overall F1 Score : {overall_f1:.4f}")

    return all_outputs, all_labels



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config") 
    parser.add_argument("--mode", type=str, default="test") 
    return parser.parse_args()


def main():
    args = get_args()
    config = read_config_file(args.config_path)
    print("Running on config {}".format(args.config_path))
    print("Mode: {}".format(args.mode))

    mode = args.mode

    test_dataset = CallGraphDataset(config, "test")
    print("Dataset has {} test samples".format(len(test_dataset)))

    if mode == "test_gpt4":
        do_test_gpt4(config, test_dataset)
    else:
        raise NotImplemented

if __name__ == '__main__':
    main()
