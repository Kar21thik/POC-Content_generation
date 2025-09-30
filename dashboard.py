# dashboard.py
import streamlit as st
import pandas as pd
import json

# We assume your QC agent code is in a file named qc_agent.py
from Validate_agent import validate_large_text 

# --- App Layout ---
st.title("📄 Text Quality Control Dashboard")

st.write("Paste a large block of text below to validate it chunk by chunk.")

# Input text area
big_text = st.text_area("Enter Text Here", height=300)

# Validate button
if st.button("Validate Text"):
    if not big_text.strip():
        st.warning("Please enter some text to validate.")
    else:
        # Run your QC agent on the input text
        validation_results = validate_large_text(big_text)
        
        # --- Display Results ---
        st.header("Validation Results")

        # Create a DataFrame for easy display
        processed_results = []
        for r in validation_results:
            result_data = r['result']
            processed_results.append({
                'Chunk': r['chunk'],
                'Is Valid': result_data.get('is_valid', False),
                'Issues': ', '.join(result_data.get('issues', [])),
                'Length Score': result_data.get('scores', {}).get('length'),
                'Structure Score': result_data.get('scores', {}).get('structure'),
                'Formatting Score': result_data.get('scores', {}).get('formatting'),
                'Profanity Score': result_data.get('scores', {}).get('profanity'),
            })
        
        df = pd.DataFrame(processed_results)

        # Display metrics for a quick overview
        total_chunks = len(df)
        invalid_chunks = len(df[df['Is Valid'] == False])
        st.metric(label="Total Chunks", value=total_chunks)
        st.metric(label="Invalid Chunks", value=invalid_chunks)

        # Display the detailed results in a table
        st.dataframe(df)

        # Create a bar chart for the average scores
        st.header("Average Quality Scores")
        avg_scores = df[['Length Score', 'Structure Score', 'Formatting Score', 'Profanity Score']].mean().dropna()
        st.bar_chart(avg_scores)
        