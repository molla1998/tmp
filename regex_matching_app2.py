import streamlit as st
import pandas as pd
import re

@st.cache_data
def load_data(csv_file="category_prefix_regex.csv"):
    return pd.read_csv(csv_file)

# Function to find the most specific prefix match
def find_matching_categories(product_id, df):
    # Sort categories by regex length (longer = more specific first)
    df = df.sort_values(by=["Prefix Regex"], key=lambda x: x.str.len(), ascending=False)

    matched_categories = []
    max_length = 0  # Track the longest matching regex length

    for _, row in df.iterrows():
        category = row["Category"]
        prefix_regex = row["Prefix Regex"]

        if re.match(prefix_regex, product_id):
            regex_length = len(prefix_regex)

            # If we find a longer (more specific) match, reset the list
            if regex_length > max_length:
                matched_categories = [category]
                max_length = regex_length
            elif regex_length == max_length:
                matched_categories.append(category)

    return matched_categories

# Streamlit UI
st.title("🔍 Product Category Finder (Most Specific Match First)")

uploaded_file = st.file_uploader("Upload your category prefix CSV file", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.warning("Please upload a valid CSV file with 'Category' and 'Prefix Regex' columns.")
    st.stop()

product_id = st.text_input("Enter a Product ID:")

if st.button("Find Category"):
    if not product_id.strip():
        st.error("Please enter a valid Product ID!")
    else:
        matches = find_matching_categories(product_id, df)

        if matches:
            st.success("✅ Most Specific Product Categories Found:")
            for match in matches:
                st.write(f"- {match}")
        else:
            st.error("❌ No matching category found.")
