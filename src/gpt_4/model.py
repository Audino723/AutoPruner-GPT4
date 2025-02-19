from dotenv import load_dotenv
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
from openai import OpenAI
import os
import tiktoken
import re
import json


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

# Initialize GPT-4 API
load_dotenv()

class GPT4_Model():
    def __init__(self, model_name="gpt-4o-mini"):
        # You can include any other initializations if needed
        self.model_name = model_name
        self.total_usage_tokens = 0
        self.total_cost = 0
        self.cost_per_1m_tokens = 0.3
        
        self.client = OpenAI(            
            api_key=os.getenv('OPENAI_API_KEY'),  # This is the default and can be omitted
        )

    def forward(self, src, dest, struct):
        # This is where you can process the code and structure if needed
        prompt = self.gpt4_pruning_prompt(src, dest, struct)
        gpt4_output = self.gpt4_predict(prompt)
        return gpt4_output, None
    
    
    def forward_cot(self, src, dest, struct):
        # This is where you can process the code and structure if needed
        prompt = self.gpt4_pruning_prompt_cot(src, dest, struct)
        gpt4_output, response = self.gpt4_predict_cot(prompt)
        return gpt4_output, response


    def gpt4_pruning_prompt(self, src, dest, struct):
        """
        Create a prompt for GPT-4 to perform call graph pruning.
        """

        # Format the tensor values into a readable string with labels
        struct_feat = "     \n".join([f"{label}: {value}" for label, value in zip(header_names, struct)])
        
        
        # Construct the refined prompt
        prompt = f"""
Analyze the following call graph edge and decide if it should be pruned or kept based on the following criteria:

1. Relevance: Does the callee’s function directly contribute to the caller’s task?
2. Kept all edges containing '<boot>' caller

Give the answer using 1 or 0 (1: Keep/0: Prune) and no other output
    
Callee Code:
{src}

Caller Code:
{dest}

Decision:
1/0
"""
        return prompt

    def gpt4_pruning_prompt_cot(self, src, dest, struct):
        """
        Create a prompt for GPT-4 to perform call graph pruning.
        """

        # Format the tensor values into a readable string with labels
        struct_feat = "     \n".join([f"{label}: {value}" for label, value in zip(header_names, struct)])
        
        
        # Construct the refined prompt
        prompt = f"""
Analyze the following call graph edge and decide if it should be pruned or kept based on the following criteria:

1. Relevance: Does the callee’s function directly or indirectly contribute to the caller’s functionality?
   - Analyze the caller's code to identify all explicit function calls and match them with the callee.
   - Consider potential indirect calls (e.g., callbacks, dynamic dispatch, higher-order functions).

2. Structure: Check the graph properties:
   - `depth_from_main`, `fanout`, `in_deg`, `out_deg`
   - Use these to determine the importance of the edge in the overall graph (e.g., prioritize edges with high `in_deg` or low `depth_from_main`).

3. Exceptions: Always keep all edges where the caller contains the term `<boot>`.

**Instructions**:
1. Perform a step-by-step analysis with a maximum of 3 essential points.
2. Use reasoning to support your decision.
3. Provide the final decision as `1` (Keep) or `0` (Prune).

**Input**:
Callee Code:
{src}

Caller Code:
{dest}

Structure:
{struct_feat}

**Reasoning**:
Step 1: Identify function calls in the caller and check for direct invocation of the callee.
Step 2: Analyze indirect invocation patterns (if no direct calls are found).
Step 3: Evaluate the structural properties and check for exceptions (`<boot>`).

---

**END OF RESPONSE**

**Decision**:
1/0
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
                # {"role": "system", "content": "You are a 10x developer that understand coding deeply."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )


        self.total_usage_tokens += response.usage.total_tokens
        self.total_cost += response.usage.total_tokens / 1e6 * self.cost_per_1m_tokens

        return int(response.choices[0].message.content) 
    
    def gpt4_predict_cot(self, prompt):
        """
        Send the prompt to GPT-4 and retrieve the response.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a call graph analysis assistant."},
                # {"role": "system", "content": "You are a 10x developer that understand coding deeply."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "pruning_reasoning",
                "schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "explanation": {"type": "string"},
                                    "output": {"type": "string"}
                                },
                                "required": ["explanation", "output"],
                                "additionalProperties": False
                            }
                        },
                        "final_answer": {"type": "string"}
                    },
                    "required": ["steps", "final_answer"],
                    "additionalProperties": False
                },
                "strict": True
                }
            }
        )

        # print(response.choices[0].message.content)

        self.total_usage_tokens += response.usage.total_tokens
        self.total_cost += response.usage.total_tokens / 1e6 * self.cost_per_1m_tokens

        response_message = json.loads(response.choices[0].message.content)
        return int(response_message['final_answer']), response_message
    
    def estimate_cost(self, input_text):
        """
        Estimate the cost for processing the input_text using OpenAI's tokenizer.
        """
        # Tokenize the input text using OpenAI's tokenizer
        encoding = tiktoken.encoding_for_model(self.model_name)
        token_count = len(encoding.encode(input_text))

        cost = (token_count / 1e6) * self.cost_per_1m_tokens

        return cost
