import os
import spacy
import torch
import docx
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Load resumes
def load_resumes(resume_directory):
    resumes = []
    resume_files = os.listdir(resume_directory)
    
    for file in resume_files:
        file_path = os.path.join(resume_directory, file)
        
        if file.endswith('.pdf'):
            resumes.append(read_pdf(file_path))  
        elif file.endswith('.docx'):
            resumes.append(read_docx(file_path))  
            
    return resumes

# Load JDs
def load_jds(jd_directory):
    jds = []
    jd_files = [file for file in os.listdir(jd_directory) if file.endswith('.xlsx')]
    
    for file in jd_files:
        file_path = os.path.join(jd_directory, file)
        
        # Read the Skills column from the Excel file
        jd_data = pd.read_excel(file_path, usecols=[0])
        skills_column = []
        for skill in jd_data.iloc[:, 0]: 
            if pd.isna(skill):
                break
            skills_column.append(skill)

        jds.append(skills_column)
        
    return jds


# PDF reader function
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# DOCX reader function
def read_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Using spacy to extract keywords
def extract_keywords(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return keywords

# Generating embeddings using BERT
def generate_bert_embeddings(keywords):
    if not keywords:
        return None
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Create tokens
    encoded_input = tokenizer(keywords, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    
    embeddings = output.last_hidden_state.mean(dim=1)  # Generate vector from all embeddings
    return embeddings

# Using cosine similarity for comparison
def calculate_similarity(doc1, doc2):
    return cosine_similarity(doc1, doc2)

def main():
    resume_directory = "./resume"
    jd_directory = "./JD" 
    
    # Load resumes and job descriptions
    resumes = load_resumes(resume_directory)
    jds_list = load_jds(jd_directory)
    
    similarity_matrix = []
    
    for resume in resumes:
        resume_keywords = extract_keywords(resume)
        
        # Generate embeddings for resume keywords
        resume_embedding = generate_bert_embeddings(resume_keywords)        
        resume_score = []
        
        # Compare with each JD file's skills
        for jd_skills in jds_list:
            processed_skills = [word for skill in jd_skills for word in skill.split()]
            
            # Generate embeddings for the processed skills
            print(processed_skills)
            jd_embedding = generate_bert_embeddings(processed_skills)
            
            if jd_embedding is None:
                print("No valid keywords found in JD.")
                continue
            
            # Calculate similarity
            similarity_score = calculate_similarity(resume_embedding, jd_embedding)
            resume_score.append(similarity_score.flatten()) 
            
        if resume_score:
            similarity_matrix.append(resume_score)
            
    # Normalize and print final similarity score
    print(similarity_matrix)
    if similarity_matrix:
        similarity_matrix = np.array(similarity_matrix)
        total_sum = np.nansum(similarity_matrix)
        
        # count valid entries 
        length = np.count_nonzero(~np.isnan(similarity_matrix))  
        if length > 0:
            result = total_sum / length
        else:
            result = np.nan
    else:
        result = np.nan  # handle case where similarity matrix is empty

    if not np.isnan(result):
        print(f"\nFinal Normalized Similarity Score: {result:.6f}")
    else:
        print("\nNo valid similarity score could be calculated.")

if __name__ == "__main__":
    main()
