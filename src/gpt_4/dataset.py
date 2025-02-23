import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.utils.utils import load_code
from src.utils.utils import read_config_file
from dgl.data.utils import makedirs, save_info, load_info
from src.utils.converter import convert
from tqdm import tqdm
import torch

# Static feature headers for graph-based information
header_names = [
    "-direct#depth_from_main", "-direct#src_node_in_deg", "-direct#dest_node_out_deg", 
    "-direct#dest_node_in_deg", "-direct#src_node_out_deg", "-direct#repeated_edges", 
    "-direct#fanout", "-direct#graph_node_count", "-direct#graph_edge_count", 
    "-direct#graph_avg_deg", "-direct#graph_avg_edge_fanout", "-trans#depth_from_main", 
    "-trans#src_node_in_deg", "-trans#dest_node_out_deg", "-trans#dest_node_in_deg", 
    "-trans#src_node_out_deg", "-trans#repeated_edges", "-trans#fanout", 
    "-trans#graph_node_count", "-trans#graph_edge_count", "-trans#graph_avg_deg", 
    "-trans#graph_avg_edge_fanout"
]

def compute_header(header_names, header):
    return [header + header_names[i] for i in range(len(header_names))]

class CallGraphDataset(Dataset):
    def __init__(self, config, mode):
        self.mode = mode
        self.config = config
        self.raw_data_path = self.config["BENCHMARK_CALLGRAPHS"]
        self.processed_path = self.config["PROCESSED_DATA"]
        self.save_dir = self.config["GPT4_CACHE_DIR"]
        self.save_path = os.path.join(self.save_dir, f"{self.mode}.pkl")
        self.cg_file = self.config["FULL_FILE"]

        if self.mode == "train":
            self.program_lists = os.path.join(self.config["TRAINING_PROGRAMS_LIST"])
        elif self.mode == "test":
            self.program_lists = os.path.join(self.config["TEST_PROGRAMS_LIST"])
        else:
            return NotImplemented

        self.header_names = compute_header(header_names, self.config["HEADERS"])

        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        struct_feats = np.where(self.struct_feats[index] == 1000000000, 100000, self.struct_feats[index])
        return {
            'src': self.data[index][0],  # Raw source (method) name
            'dst': self.data[index][1],  # Raw destination (target) name
            'struct': torch.tensor(struct_feats, dtype=torch.float),  # Static features (in/out degrees, etc.)
            'label': torch.tensor(self.labels[index], dtype=torch.long),  # Label
            'static_ids': self.static_ids[index],
        }

    def process(self):
        self.data = []
        self.struct_feats = []
        self.labels = []
        self.static_ids = []

        with open(self.program_lists, "r") as f:
            for line in tqdm(f):
                filename = line.strip()
                file_path = os.path.join(self.raw_data_path, filename, self.cg_file)
                df = pd.read_csv(file_path)

                features = df[self.header_names].to_numpy()
                for i in tqdm(range(len(df['wiretap']))):
                    src, dst, lb, sanity_check = df['method'][i], df['target'][i], df['wiretap'][i], df[self.config["SA_LABEL"]][i]
                    
                    descriptor2code = load_code(os.path.join(self.processed_path, filename, 'code.csv'), new_line=True)
                    if src != '<boot>':
                        if src in descriptor2code:
                            src = descriptor2code[src]
                        else:
                            src = convert(src).__tocode__()
                    
                    dst_descriptor = convert(dst)
                    
                    if dst in descriptor2code:
                        dst = descriptor2code[dst]
                    else:
                        dst = dst_descriptor.__tocode__()

                    # Extract the static features for the call graph
                    self.data.append([src, dst])
                    self.struct_feats.append(features[i])
                    self.labels.append(lb)
                    self.static_ids.append(sanity_check)


    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        save_info(self.save_path, {
            'data': self.data,
            'struct_feats': self.struct_feats,
            'labels': self.labels,
            'static_ids': self.static_ids
        })

    def load(self):
        print("Loading data ...")
        info_dict = load_info(self.save_path)
        self.data = info_dict['data']
        self.struct_feats = info_dict['struct_feats']
        self.labels = info_dict['labels']
        self.static_ids = info_dict['static_ids']

    def has_cache(self):
        if os.path.exists(self.save_path):
            print("Data exists")
            return True
        return False

if __name__ == '__main__':
    config = read_config_file("config/wala.config")
    train_data = CallGraphDataset(config, "train")
    test_data = CallGraphDataset(config, "test")
