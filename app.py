import os
import spacy
import torch
import docx
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load resumes from the directory (support for PDF and DOCX)
def load_resumes(resume_directory):
    resumes = []
    resume_files = os.listdir(resume_directory)
    
    # Loop through all resume files
    for file in resume_files:
        file_path = os.path.join(resume_directory, file)
        if file.endswith('.pdf'):
            resumes.append(read_pdf(file_path))  # Indented
        elif file.endswith('.docx'):
            resumes.append(read_docx(file_path))  # Indented
    return resumes  # Indented


# Step 2: Load JDs from the directory (support for PDF and DOCX)
def load_jds(jd_directory):
    jds = []
    jd_files = os.listdir(jd_directory)
    
    for file in jd_files:
        file_path = os.path.join(jd_directory, file)
        if file.endswith('.pdf'):
            jds.append(read_pdf(file_path))
        elif file.endswith('.docx'):
            jds.append(read_docx(file_path))
    return jds

# Function to read text from PDF
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to read text from DOCX
def read_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Step 3: Extract keywords from text using SpaCy
def extract_keywords(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return keywords

# Step 4: Generate embeddings using BERT
def generate_bert_embeddings(keywords):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Encode keywords using BERT tokenizer
    encoded_input = tokenizer(keywords, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)

    # Output the last hidden states
    embeddings = output.last_hidden_state.mean(dim=1)
    return embeddings

# Step 5: Calculate Cosine Similarity between two sets of embeddings
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)

# Main function to compare resumes and JDs
def main():
    resume_directory = "./resume"  # Directory where resumes are stored
    jd_directory = "./JD"  # Directory where job descriptions are stored
    
    # Load resumes and job descriptions
    resumes = load_resumes(resume_directory)
    jds = load_jds(jd_directory)
    
    i = 0
    for resume in resumes:
        resume_keywords = extract_keywords(resume)
        resume_embedding = generate_bert_embeddings(resume_keywords)
        
        # print(f"\nResume:\n{resume[:100]}...")  # Display first 100 chars of the resume for context

        # Compare with each JD
        j = 0
        for jd in jds:
            jd_keywords = extract_keywords(jd)
            jd_embedding = generate_bert_embeddings(jd_keywords)
            
            similarity_score = calculate_similarity(resume_embedding, jd_embedding)
            print(similarity_score)
            print(f"Similarity Score with JD: {similarity_score[i][j]:.4f}")
            # j += 1
        # i += 1

if __name__ == "__main__":
    main()
