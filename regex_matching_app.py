import streamlit as st
import pandas as pd
import re

# Load CSV file with category, prefix, and suffix regex
@st.cache_data
def load_data(csv_file="category_prefix_suffix_regex.csv"):
    return pd.read_csv(csv_file)

# Function to find matching categories
def find_matching_categories(product_id, df):
    matching_categories = set()

    for _, row in df.iterrows():
        category = row["Category"]
        prefix_regex = row["Prefix Regex"]
        suffix_regex = row["Suffix Regex"]

        # Match Prefix and Suffix
        prefix_match = re.match(prefix_regex, product_id)
        suffix_match = re.match(suffix_regex, product_id)

        if prefix_match and suffix_match:
            matching_categories.add(f"{category} (Prefix & Suffix Match)")
        elif prefix_match:
            matching_categories.add(f"{category} (Prefix Match)")
        elif suffix_match:
            matching_categories.add(f"{category} (Suffix Match)")

    return list(matching_categories)

# Streamlit UI
st.title("🔍 Product Category Finder")

# Upload CSV File
uploaded_file = st.file_uploader("Upload your category prefix-suffix CSV file", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.warning("Please upload a valid CSV file with 'Category', 'Prefix Regex', and 'Suffix Regex' columns.")
    st.stop()

# User Input for Product ID
product_id = st.text_input("Enter a Product ID:")

# Process on Button Click
if st.button("Find Category"):
    if not product_id.strip():
        st.error("Please enter a valid Product ID!")
    else:
        # Find matching categories
        matches = find_matching_categories(product_id, df)

        # Display Results
        if matches:
            st.success("✅ Possible Product Categories Found:")
            for match in matches:
                st.write(f"- {match}")
        else:
            st.error("❌ No matching category found.")

# Footer
st.markdown("---")
st.markdown("🚀 **Developed with Streamlit & Regex** | 💡 Supports overlapping matches!")
