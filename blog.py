import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Hugging Face token from .env
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load model and tokenizer from Hugging Face using API token
def load_llama_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", use_auth_token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", use_auth_token=HUGGINGFACE_TOKEN)


    # Setup pipeline for text generation
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    return generator

# Function to generate response from Llama model
def get_llama_response(generator, input_text, no_words, blog_style):
    prompt = f"Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words."
    response = generator(prompt, max_length=int(no_words), do_sample=True, temperature=0.7)
    return response[0]['generated_text']


# Streamlit app
st.set_page_config(page_title="Generate Blogs 🤖", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')
st.header("Generate Blogs 🤖")

# Input fields
input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')

with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

# Generate the blog if submit is clicked
if submit:
    generator = load_llama_model()
    response = get_llama_response(generator, input_text, no_words, blog_style)
    st.write(response)
