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
    def __init__(self, model_name="gpt-4o"):
        # You can include any other initializations if needed
        self.model_name = model_name
        self.total_usage_tokens = 0
        self.total_cost = 0
        self.cost_per_1m_tokens = 0.3
        self.temperature = 0.0
        
        self.client = OpenAI(            
            api_key=os.getenv('OPENAI_API_KEY'),  # This is the default and can be omitted
        )

    def forward(self, src, dest, struct):
        # This is where you can process the code and structure if needed
        prompt = self.gpt4_pruning_prompt(src, dest, struct)
        gpt4_output = self.gpt4_predict(prompt)
        return gpt4_output, None, prompt
    
    
    def forward_cot(self, src, dest, struct):
        # This is where you can process the code and structure if needed
        prompt = self.gpt4_pruning_prompt_cot(src, dest, struct)
        gpt4_output, response = self.gpt4_predict_cot(prompt)
        return gpt4_output, response, prompt


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
    
Caller Code:
{dest}

Callee Code:
{src}

Decision:
1/0
"""
        
        
        prompt = f"""
I have constructed a call graph using static analysis, and I need your help to prune irrelevant edges. For each edge, I will provide the following information:

1. Caller Code: The function or block of code attempting the call.
2. Callee Candidates: One or more possible target functions for the call.

Your task is to determine whether the edge between the caller and each callee candidate should be kept or pruned. Follow these steps:
1. Analyze the Semantic Relationship:
   - Check if the caller and callee are logically connected based on their functionality and purpose.
   - Prune the edge if the callee's functionality does not align with the caller’s context or intent.
2. Utility Filtering:
   - Identify generic or utility functions (e.g., print(), log()) that are commonly used but not central to the logic.
   - Prune such edges unless they are directly relevant to the caller’s purpose.
3. Dynamic Dispatch:
   - For polymorphic or dynamic calls, analyze the caller’s context to determine which callee candidates could realistically be invoked.
   - Keep only the valid candidates and prune the rest.
4. Special Case - <boot> Callees:
   - <boot> callees often represent system-generated, default, or framework-specific initialization functions.
   - Keep <boot> callees only if:
      - The caller logically requires initialization or bootstrapping (e.g., startup processes, default handling).
      - The <boot> callee is critical to the execution flow. Otherwise, prune <boot> callees as irrelevant noise.
5. Optional - Use Static Features as a Final Gate:
    If the semantic analysis is inconclusive, use provided static features (e.g., in-degree, out-degree) to refine your decision:
        1. Prune edges with very high in-degree if they are likely noise (e.g., utility functions).
        2. Retain edges for nodes with low in-degree or out-degree if they seem critical to the flow (e.g., leaf nodes or controllers).

**Input**:

Caller Code:
{dest}

Callee Code:
{src}

Structure:
{struct_feat}

Response Format:
Give the answer using 1 or 0 (1: Keep/0: Prune) and no other output
"""
        return prompt

    def gpt4_pruning_prompt_cot(self, src, dest, struct):
        """
        Create a prompt for GPT-4 to perform call graph pruning.
        """

        # Format the tensor values into a readable string with labels
        struct_feat = ",\n    ".join([f'"{label}": {value}' for label, value in zip(header_names, struct)])
        
        prompt = f"""
I have constructed a call graph using static analysis, and I need your help to prune irrelevant edges. For each edge, I will provide the following information:

1. Caller Code: The function or block of code attempting the call.
2. Callee Candidates: One or more possible target functions for the call.

Your task is to determine whether the edge between the caller and each callee candidate should be kept or pruned. Follow these steps:
1. Special Case - <boot> Callees:
   - <boot> callees often represent system-generated, default, or framework-specific initialization functions. Keep it.
2. Analyze the Semantic Relationship:
   - Check if the caller and callee are logically connected based on their functionality and purpose.
   - Prune the edge if the callee's functionality does not align with the caller’s context or intent.
3. Utility Filtering:
   - Identify generic or utility functions (e.g., print(), log()) that are commonly used but not central to the logic.
   - Prune such edges unless they are directly relevant to the caller’s purpose.
4. Dynamic Dispatch:
   - For polymorphic or dynamic calls, analyze the caller’s context to determine which callee candidates could realistically be invoked.
   - Keep only the valid candidates and prune the rest. 
5. Analyze the Directionality:
   - Identify which code block is the caller and which is the callee by verifying their parent function names.
   - Validate whether the caller explicitly calls the callee. The edge should only be kept if the caller directly invokes the callee within its logic.
   - If the callee invokes the caller or the relationship is undefined, prune the edge.
6. Analyze Static Features:
    If the semantic analysis is inconclusive, use provided static features (e.g., in-degree, out-degree) to refine your decision:
        1. Prune edges with very high in-degree if they are likely noise (e.g., utility functions).
        2. Retain edges for nodes with low in-degree or out-degree if they seem critical to the flow (e.g., leaf nodes or controllers).

**Input**:

**Caller Code**:
```java
{dest}
```

**Callee Code**:
```java
{src}
```

**Structure**:
```json
{struct_feat}
```

Response Format:
Provide the final decision as `1` (Keep) or `0` (Prune).
"""
        
        
        prompt = f"""
Given an edge from a call-graph constructed by a WALA static analysis, determine wether its a true positive or a false positive.    

**Input**:

**Caller Code**:
```java
{dest}
```

**Callee Code**:
```java
{src}
```

**Structure**:
```json
{struct_feat}
```

Response Format:
Give your explanation in one sentence.
Provide the final decision as `1` (TP) or `0` (FP).
"""

        prompt = """
ewr
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
            temperature=self.temperature,
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
            temperature=self.temperature,
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
