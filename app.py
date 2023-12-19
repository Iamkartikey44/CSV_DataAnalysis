import streamlit as st
from dotenv import load_dotenv
from utils import query_agent

st.title("Let's do some analysis on your CSV")
st.header("Please upload your CSV file here:")

data = st.file_uploader("Upload CSV File",type='csv')

query = st.text_area("Enter your query")
button = st.button("Generate Response")

if button:
    answer = query_agent(data,query)
    st.write(answer)
