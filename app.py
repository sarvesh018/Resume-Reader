import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import BertTokenizer, BertModel
import torch
import docx
from PyPDF2 import PdfReader

# Loading Resume
def load_resume(resume_directory):
    resume_files = os.listdir(resume_directory)
    
    # Check for a valid resume .pdf or .docx
    for file in resume_files:
        file_path = os.path.join(resume_directory, file)
        
        if file.endswith('.pdf'):
            return read_pdf(file_path)
        elif file.endswith('.docx'):
            return read_docx(file_path)
    raise FileNotFoundError("No valid resume files (PDF/DOCX) found in the directory.")

# Function to read text from PDF document
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to read text from DOCX document
def read_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Using SpaCy library for extracting keywords from resume
def extract_keywords(resume_text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(resume_text)

    # Lemmatization
    keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return keywords

# Generate embeddings or feature vectors using BERT
def generate_bert_embeddings(keywords):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Encode keywords using BERT tokenizer
    encoded_input = tokenizer(keywords, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
        
    embeddings = output.last_hidden_state.mean(dim=1)
    return embeddings
    
# main function
def main():
    resume_directory = "./resume"
    resume_content = load_resume(resume_directory)
    print("Loaded Resume Content")

    keywords = extract_keywords(resume_content)
    print(f"Extracted Keywords: {keywords}")

    embeddings = generate_bert_embeddings(keywords)
    print(f"Generated BERT Embeddings: {embeddings}")

if __name__ == "__main__":
    main()
