**AI-Powered Resume Screening System**

**Overview**

The AI-powered Resume Screening System is a Streamlit-based web application that automates resume screening, ranking, and interview question generation. The system extracts text from resumes, predicts job categories, ranks resumes based on job descriptions, and generates dynamic interview questions using TF-IDF and cosine similarity.

**Features**

    **Resume Parsing**: Extracts text from PDF, DOCX, and TXT resumes.

    **Category Prediction**: A trained SVM model is used to classify resumes into job categories.

    **Resume Ranking**: Employs TF-IDF & cosine similarity to rank resumes against a job description.

    **Interview Question Generation**: Dynamically generates technical & behavioural questions.

    **User-Friendly Interface**: Built with Streamlit for an interactive experience.

**Installation**

1. Clone the Repository

        git clone https://github.com/your-username/resume-screening.git
        cd resume-screening

2. Install Dependencies

        pip install -r requirements.txt

3. Run the Application

        streamlit run app1.py

**Dependencies**

Ensure you have the following Python libraries installed:
   
    streamlit
    pickle
    docx
    PyPDF2
    re
    pandas
    scikit-learn

Install missing dependencies using:
     
     pip install streamlit docx PyPDF2 pandas scikit-learn

**How It Works**

**Step 1**: Upload Resumes
    Upload multiple resumes in PDF, DOCX, or TXT format.

**Step 2**: Enter Job Description
    Provide the job description text in the input box.

**Step 3**: Resume Analysis
    The system extracts text from resumes.
    Each resume is categorized based on its content.
    Resumes are ranked based on their relevance to the job description.

**Step 4**: Generate Interview Questions
    Extracts keywords from the resume using TF-IDF.
    Generates dynamic interview questions based on the candidate's experience.

**Project Structure**

    resume-screening/
        │── models/                # Pre-trained models (SVM classifier, TF-IDF, Label Encoder)
        │── app1.py                # Main application file
        │── requirements.txt        # List of dependencies
        │── README.md               # Project documentation

**Future Enhancements**
   
Enhanced NLP models for better classification.

ATS Integration to streamline recruitment.

Dashboard for visualization of resume statistics.
