from dotenv import load_dotenv
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
from openai import OpenAI
import os
import tiktoken


# Initialize GPT-4 API
load_dotenv()

class GPT4_Model():
    def __init__(self, model_name="gpt-4o"):
        # You can include any other initializations if needed
        self.model_name = model_name
        
        self.client = OpenAI(            
            api_key=os.getenv('OPENAI_API_KEY'),  # This is the default and can be omitted
        )

    def forward(self, code, struct):
        # This is where you can process the code and structure if needed
        prompt = self.gpt4_pruning_prompt(code, struct)
        gpt4_output = self.gpt4_predict(prompt)
        return gpt4_output

    def gpt4_pruning_prompt(self, src, dest, struct):
        """
        Create a prompt for GPT-4 to perform call graph pruning.
        """
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
        
        # Construct the refined prompt
        prompt = f"""The following is raw data for a program call graph:
        Callee Code: 
        {dest}

        Caller Code:
        {src}

        Structure: 
        {struct}

        The structure includes the following graph properties:
        {', '.join(header_names)}

        The graph properties have been categorized into two groups: 
        - Direct properties (e.g., in-degrees, out-degrees, fanout) describe the immediate characteristics of nodes.
        - Transitive properties describe how nodes relate indirectly through other nodes (e.g., depth from the main node, average graph degree).

        The goal is to prune unnecessary nodes and edges while maintaining the essential functionality of the graph.
        
        In your pruning process:
        1. Identify nodes with low degrees (both in and out), as they may not significantly impact the overall structure.
        2. Consider removing nodes with high fanout if they do not contribute to the main logic or functionality.
        3. Nodes with repeated edges or low graph metrics might be less important and can be candidates for removal.
        4. Ensure that the pruned graph retains its core structure, meaning the key relationships between nodes should remain intact, and important program functionality should not be lost.

        Provide the pruned graph structure as output. Be sure to include an explanation of why each node or edge was removed based on the provided features.
        """
        
        return prompt

    def gpt4_predict(self, prompt):
        """
        Send the prompt to GPT-4 and retrieve the response.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a call graph analysis assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response['choices'][0]['message']['content']

    def parse_gpt4_output(self, gpt4_output):
        """
        Parse the GPT-4 output into a format suitable for evaluation.
        """
        # Example: Assume the output is a string of pruned node IDs
        try:
            pred = list(map(int, gpt4_output.strip().split(',')))  # Adjust parsing logic as needed
        except Exception as e:
            print(f"Error parsing GPT-4 output: {e}")
            pred = []
        return pred
    
    def estimate_cost(self, input_text):
        """
        Estimate the cost for processing the input_text using OpenAI's tokenizer.
        """
        # Tokenize the input text using OpenAI's tokenizer
        encoding = tiktoken.encoding_for_model("gpt-4o")
        token_count = len(encoding.encode(input_text))

        cost_per_million_tokens = 0.3  # Example cost in USD per million tokens
        cost = (token_count / 1e6) * cost_per_million_tokens

        return cost
