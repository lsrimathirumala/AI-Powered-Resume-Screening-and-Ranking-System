import streamlit as st
import pickle
from docx import Document  
import PyPDF2 
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load trained models
with open('models/clf.pkl', 'rb') as model_file:
    svc_model = pickle.load(model_file)

with open('models/tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

with open('models/encoder.pkl', 'rb') as encoder_file:
    le = pickle.load(encoder_file)

# Function to clean and preprocess resume text
def clean_resume_text(text):
    text = re.sub('http\S+\s', ' ', text)  
    text = re.sub('RT|cc', ' ', text) 
    text = re.sub('#\S+\s', ' ', text)  
    text = re.sub('@\S+', ' ', text)  
    text = re.sub('[%s]' % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), ' ', text) 
    text = re.sub(r'[^\x00-\x7f]', ' ', text)  
    text = re.sub('\s+', ' ', text)  
    return text.strip()

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + '\n'
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(file):
    doc = Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from a TXT file
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')  
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')  
    return text

# Function to process file upload
def process_uploaded_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

# Function to predict resume category
def predict_category(resume_text):
    cleaned_text = clean_resume_text(resume_text)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

# Function to rank resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Function to generate interview questions dynamically
def generate_questions(resume_text, num_questions):
    # Extract keywords using TF-IDF
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text])
    keywords = vectorizer.get_feature_names_out()

    # Question templates
    question_templates = [
        "Can you explain your experience with {}?",
        "What challenges have you faced while working on {}?",
        "How would you improve your skills in {}?",
        "Describe a project where you used {} effectively.",
        "How do you stay updated with the latest advancements in {}?",
        "Tell me about a mistake you made related to {} and how you fixed it.",
        "What are the most common issues in {} and how do you handle them?",
        "How would you apply {} in a real-world project?",
        "What is the most complex problem you've solved using {}?",
        "How would you teach {} to someone new to the field?"
    ]

    questions = []

    # Generate questions dynamically
    for keyword in keywords:
        template = random.choice(question_templates)  # Pick a random template
        question = template.format(keyword)  # Fill it with a keyword
        questions.append(question)

    # Behavioral and Situational Questions
    behavioral_templates = [
        "Tell me about a time you faced a challenge at work and how you overcame it.",
        "Describe a situation where you had to work in a team to achieve a goal.",
        "If you were given a tight deadline for a complex project, how would you handle it?",
        "Imagine you're leading a project, but a key team member is unavailable. What would you do?",
        "Tell me about a time when you had to quickly learn a new technology or skill."
    ]

    # Add random behavioral questions
    questions += random.sample(behavioral_templates, min(len(behavioral_templates), num_questions // 2))

    # Ensure we return exactly num_questions
    return random.sample(questions, min(len(questions), num_questions))

# Streamlit App
def main():
    st.set_page_config(page_title="Resume Screening", page_icon="📄", layout="wide")
    st.title("AI-Powered Resume Screening System")
    
    job_description = st.text_area("Enter the job description")
    
    uploaded_files = st.file_uploader("Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    
    if uploaded_files and job_description:
        resumes = [process_uploaded_file(file) for file in uploaded_files]
        
        # Predict categories
        categories = [predict_category(text) for text in resumes]
        
        # Rank resumes
        scores = rank_resumes(job_description, resumes)
        results = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files], 
            "Category": categories,
            "Score": scores
        })
        results = results.sort_values(by="Score", ascending=False)
        
        st.subheader("Ranked Resumes")
        st.write(results)
        
        # Display extracted text & interview questions
        for file, text in zip(uploaded_files, resumes):
            st.subheader(f"Extracted Text from {file.name}")
            st.text_area("Extracted Resume Text", text, height=200)

            # User input: Number of questions
            num_questions = st.number_input(f"Select number of questions for {file.name}", min_value=1, max_value=10, value=5)

            # Button to generate questions
            if st.button(f"Generate Questions for {file.name}"):
                questions = generate_questions(text, num_questions)
                st.subheader("Interview Questions")
                for q in questions:
                    st.write(f"- {q}")

if __name__ == "__main__":
    main()
