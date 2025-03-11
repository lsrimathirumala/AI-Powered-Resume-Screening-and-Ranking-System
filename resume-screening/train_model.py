import os
import pickle
import PyPDF2
from docx import Document
import re
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Function to extract text from different file types
def extract_text(file_path):
    ext = file_path.split('.')[-1].lower()
    text = ""
    try:
        if ext == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        elif ext == "docx":
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return clean_text(text)

# Function to categorize resumes based on keywords
def categorize_resume(text):
    text = text.lower()
    if "machine learning" in text or "deep learning" in text:
        return "Machine Learning Engineer"
    elif "web development" in text or "frontend" in text or "backend" in text:
        return "Web Developer"
    elif "data science" in text or "data analysis" in text:
        return "Data Scientist"
    elif "cyber security" in text or "ethical hacking" in text:
        return "Cyber Security Analyst"
    else:
        return "Other"

# Load resumes
def load_resumes():
    resumes = []
    categories = []
    
    if not os.listdir(UPLOAD_FOLDER):
        print("🚨 No resumes found in uploads/. Please upload resumes first.")
        return None, None

    for file_name in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        text = extract_text(file_path)
        
        if text:
            resumes.append(text)
            category = categorize_resume(text)  # Assign category
            categories.append(category)

    return resumes, categories

# Train model
resumes, categories = load_resumes()

if resumes is not None and categories is not None:
    unique_categories = set(categories)
    if len(unique_categories) < 2:
        print(f"🚨 Error: Only 1 category found: {unique_categories}. At least 2 categories are required for training.")
    else:
        tfidf = TfidfVectorizer()
        X_train_tfidf = tfidf.fit_transform(resumes)

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(categories)

        svc_model = SVC()
        svc_model.fit(X_train_tfidf, y_train_encoded)

        # Save model
        with open(os.path.join(MODEL_FOLDER, "clf.pkl"), "wb") as model_file:
            pickle.dump(svc_model, model_file)
        with open(os.path.join(MODEL_FOLDER, "tfidf.pkl"), "wb") as tfidf_file:
            pickle.dump(tfidf, tfidf_file)
        with open(os.path.join(MODEL_FOLDER, "encoder.pkl"), "wb") as encoder_file:
            pickle.dump(le, encoder_file)

        print(f"✅ Model training complete! Categories used: {unique_categories}")
else:
    print("⚠️ Training skipped due to no available resumes.")
