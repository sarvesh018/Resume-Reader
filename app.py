import os
import spacy
import torch
import docx
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd  # For handling Excel files

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

# Load JDs from multiple Excel files in the JD directory
def load_jds(jd_directory):
    jds = []
    jd_files = [file for file in os.listdir(jd_directory) if file.endswith('.xlsx')]
    
    for file in jd_files:
        file_path = os.path.join(jd_directory, file)
        # Read the Excel file
        jd_data = pd.read_excel(file_path, usecols=[0])  # Only load the first column
        
        # Extract skills from the first column up to the first empty cell
        skills_column = []
        for skill in jd_data.iloc[:, 0]:  # Access the first column
            if pd.isna(skill):  # Check if the cell is empty
                break  # Stop if an empty cell is found
            skills_column.append(skill)

        jds.append(skills_column)  # Append the list of skills to jds
        
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
        return None  # If no keywords, return None
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
    if doc1 is None or doc2 is None:
        return np.nan  # If any embedding is None, return NaN similarity
    return cosine_similarity(doc1, doc2)

def main():
    resume_directory = "./resume"
    jd_directory = "./JD"  # Directory containing multiple JD Excel files
    
    # Load resumes and job descriptions
    resumes = load_resumes(resume_directory)
    jds_list = load_jds(jd_directory)  # Load multiple JDs
    
    similarity_matrix = []
    
    for resume in resumes:
        # Extract keywords from the resume text
        resume_keywords = extract_keywords(resume)
        # print("\nExtracted Resume Keywords:", resume_keywords)
        
        # Generate embeddings for resume keywords
        resume_embedding = generate_bert_embeddings(resume_keywords)
        if resume_embedding is None:
            print("No valid keywords found in resume.")
            continue
        
        resume_score = []
        
        # Compare with each JD file's skills
        for jd_skills in jds_list:
            print(jd_skills)
            # jd_keywords = jd_skills.split(",")  # Assuming skills are comma-separated
            
            # Generate embeddings for JD keywords
            jd_embedding = generate_bert_embeddings(jd_skills)
            if jd_embedding is None:
                print("No valid keywords found in JD.")
                continue
            
            # Calculate similarity
            similarity_score = calculate_similarity(resume_embedding, jd_embedding)
            # print("Similarity Score:", similarity_score)
            resume_score.append(similarity_score.flatten())  # Flatten simplifies complex arrays
            
        if resume_score:  # Ensure we only append non-empty score arrays
            similarity_matrix.append(resume_score)
            
    # Normalize and print final similarity score
    print(similarity_matrix)
    if similarity_matrix:
        similarity_matrix = np.array(similarity_matrix)
        total_sum = np.nansum(similarity_matrix)  # Use nansum to handle NaN values
        length = np.count_nonzero(~np.isnan(similarity_matrix))  # Only count valid entries
        if length > 0:
            result = total_sum / length
        else:
            result = np.nan  # Set result to NaN if no valid entries
    else:
        result = np.nan  # Handle case where similarity matrix is empty

    if not np.isnan(result):
        print(f"\nFinal Normalized Similarity Score: {result:.6f}")
    else:
        print("\nNo valid similarity score could be calculated.")

if __name__ == "__main__":
    main()
