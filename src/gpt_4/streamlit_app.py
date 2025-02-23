import streamlit as st
import pandas as pd
import numpy as np

# df = pd.read_csv("../replication_package/gpt4_dataset/wala/gpt4_result_0_100.csv")

# Streamlit app
st.set_page_config(page_title="Call Graph Viewer", layout="wide")   
st.title("Call Graph Code Viewer")

file_path = st.text_input("Enter the path to your call graph CSV file", value="../replication_package/gpt4_dataset/wala/gpt4_result_0_10.csv")
df = pd.read_csv(file_path)


# Persistent state to track the current index
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Navigation buttons
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("⬅️ Previous"):
        st.session_state.current_index = max(0, st.session_state.current_index - 1)

with col3:
    if st.button("➡️ Next"):
        st.session_state.current_index = min(len(df) - 1, st.session_state.current_index + 1)

# Get the current row
current_row = df.iloc[st.session_state.current_index]

# Display the current data
st.write(f"### Viewing Code for Index: {current_row['Index']}")

# Optionally display additional information
st.write("### Additional Information")
st.write(f"**Structure:** {current_row['Structure']}")
st.write(f"**Label:** {current_row['Label']}")
st.write(f"**Prediction:** {current_row['Prediction']}")

# Display the JSON response
st.write("### Response")
st.json(current_row["Response"])  # Properly formats and displays the JSON response

# Create two columns for side-by-side comparison
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Callee Code (Start)")
    st.code(current_row["Start"], language="java")  # Format with line breaks

with col2:
    st.subheader("Caller Code")
    st.code(current_row["Destination"], language="java")  # Format with line breaks

with col3:
    st.subheader("Prompt")
    st.write(current_row["Prompt"])  # Display as plain text


