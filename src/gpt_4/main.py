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


warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PARAMS = {'batch_size': 15, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': 10, 'shuffle': False, 'num_workers': 8}

# main.py

def do_test_gpt4(dataset):
    """
    Test GPT-4's ability to prune call graphs using the dataset.
    """
    all_outputs = []
    all_labels = []
    precision_list, recall_list, f1_list = [], [], []
    estimate_cost = 0

    gpt4_model = GPT4_Model()

    print("begin testing...")
    loop = tqdm(enumerate(dataset), leave=False, total=len(dataset))
    for idx, batch in loop:
        if idx > 10000:
            break

        src = batch['src']
        dst = batch['dst']
        struct = batch['struct']
        label = batch['label'].numpy()

        prompt = gpt4_model.gpt4_pruning_prompt(src, dst, struct)
        estimate_cost += gpt4_model.estimate_cost(prompt)

        continue

        # Get GPT-4's response
        gpt4_output = gpt4_model(code, struct)

        # Parse GPT-4 output into predictions (this will depend on GPT-4's response format)
        pred = gpt4_model.parse_gpt4_output(gpt4_output)

        # Evaluate predictions
        precision = precision_score(label, pred, zero_division=0)
        recall = recall_score(label, pred, zero_division=0)
        f1 = f1_score(label, pred, zero_division=0)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # Append results for overall metrics
        all_outputs.extend(pred)
        all_labels.extend(label)

        # Update progress bar
        loop.set_postfix(pre=precision, rec=recall, f1=f1)

    # Compute overall metrics
    overall_precision = np.mean(precision_list)
    overall_recall = np.mean(recall_list)
    overall_f1 = np.mean(f1_list)

    print(f"Estimate cost     : {estimate_cost:.2f} $")
    print(f"Overall Precision : {overall_precision:.4f}")
    print(f"Overall Recall    : {overall_recall:.4f}")
    print(f"Overall F1 Score : {overall_f1:.4f}")
    return all_outputs, all_labels


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config") 
    parser.add_argument("--mode", type=str, default="test") 
    parser.add_argument("--model_path", type=str, default="../replication_package/model/rq1/autopruner/wala.pth", help="Path to checkpoint (for test only)") 
    parser.add_argument("--feature", type=int, default=2, help="0: structure, 1: semantic, 2:combine")     
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
        do_test_gpt4(test_dataset)
    else:
        raise NotImplemented

if __name__ == '__main__':
    main()
