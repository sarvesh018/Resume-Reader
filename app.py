import os
import spacy
import torch
import docx
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np

# load resume
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


# load JD
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

# pdf reader function
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# docx reader function
def read_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# using spacy to extract keyword
def extract_keywords(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return keywords



# generating embeddings using BERT
def generate_bert_embeddings(keywords):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # create tokens
    encoded_input = tokenizer(keywords, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    
    embeddings = output.last_hidden_state.mean(dim=1) # this line will generate vector from all the embbedings
    return embeddings




# using cosine similarity for comparison
def calculate_similarity(doc1, doc2):
    return cosine_similarity(doc1, doc2)




def main():
    resume_directory = "./resume"
    jd_directory = "./JD"
    
    # loading resumes and job descriptions
    resumes = load_resumes(resume_directory)
    jds = load_jds(jd_directory)
    
    similarity_matrix = []
    
    for resume in resumes:
        resume_keywords = extract_keywords(resume)
        resume_embedding = generate_bert_embeddings(resume_keywords)
        
        resume_score= []
        
        # comparing with each JD
        for jd in jds:
            jd_keywords = extract_keywords(jd)
            jd_embedding = generate_bert_embeddings(jd_keywords)
            
            similarity_score = calculate_similarity(resume_embedding, jd_embedding)
            print(similarity_score)
            resume_score.append(similarity_score.flatten())  # flatten simplifies complex arrays
            
        similarity_matrix.append(resume_score)
            
    similarity_matrix = np.array(similarity_matrix)
    total_sum = np.sum(similarity_matrix)
    length = similarity_matrix.size
    result = total_sum / length

    print(f"\nFinal Normalized Similarity Score: {result:.6f}")

if __name__ == "__main__":
    main()
