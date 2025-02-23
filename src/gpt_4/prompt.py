

        
        prompt = f"""
Use ur feeling or common assumption to determine wether the destination is reachable from the source. Could be directl or indirectly. `<boot>` can reach anywhere. 

REMEMBER FROM SOURCE TO DESTINATION. THE SOURCE IS CALLED FROM THE DESTINATION DIRECTL OR INIDIRECTLY

SOURRCE: CALLEE
DESTINATION: CALLER

**Input**:

**Source**:
```java
{src}
```

**Destination**:
```java
{dest}
```

**Structure**:
```json
{
    {struct_feat}
}
```

Response Format:
Provide the final decision as 1 (reachable) or 0 (not reachable). If the decision is based on assumptions or inferred logic, explain the reasoning.
"""
        
        prompt = f"""
You are tasked with evaluating the validity of an edge in a call graph. For the given edge, you have access to:

1. The caller function's code.
2. The callee function's code.
3. Static features, including the in-degree of the callee and the out-degree of the caller.

Follow these steps to determine if the edge is valid (reachable) or should be pruned (not reachable):
1. Retrieve Edge Details:
   - Identify the caller and callee functions.
   - Check their respective in-degree and out-degree values.
2. Analyze the Edge:
   - Direct Reference Check: Does the caller directly invoke the callee? If yes, the edge is valid. If no, continue to the next step.
   - Indirect Call Analysis:
       - Look for indirect calls through helper methods, delegation, or dynamic dispatch (e.g., polymorphism).
       - Functions with a high out-degree (caller) or high in-degree (callee) are more likely to involve indirect invocation and deserve closer inspection.
   - Contextual Clues: Inspect comments, Javadoc, or documentation in the caller or callee code for any mention of delegation or usage patterns.
3. Make a Decision:
   - If the caller directly or indirectly invokes the callee (e.g., through helper functions or runtime dispatch), the edge is valid (reachable).
   - If no evidence of direct or indirect invocation exists, the edge is not valid (not reachable).

Example:
Given a caller function PrintfFormat::sprintf and a callee function PrintfFormat::internalsprintf:
- Direct invocation is not present in the provided code.
- Delegation or hidden dependencies may exist (e.g., sprintf could internally call internalsprintf).
- Based on analysis and assumptions, this edge is reachable (1).


**Callee**:
```java
{src}
```

**Caller**:
```java
{dest}
```

**Structure**:
```json
{
    {struct_feat}
}
```

Use this approach to evaluate edges in the call graph systematically. Your final output should be 1 (reachable) or 0 (not reachable), with reasoning if applicable.
"""


        
        # Construct the refined prompt
        prompt = f"""
Analyze the following call graph edge and decide if it should be pruned or kept based on the following criteria:

1. Relevance: Does the callee’s function directly or indirectly contribute to the caller’s functionality?
   - Analyze the caller's code to identify all explicit function/class calls and match them with the callee.
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

Caller Code:
{dest}

Callee Code:
{src}

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
        