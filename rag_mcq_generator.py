import os
import time
import random
import re
import pickle
import faiss
import numpy as np
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
PDF_FILE_PATH = os.getenv("PDF_FILE_PATH")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup Gemini
MODEL_ID = "gemma-3-12b-it"
genai.configure(api_key=GOOGLE_API_KEY)

# Subject-wise subtopics
subject_subtopics = {
    "Anatomy": ["gross anatomy", "microscopic anatomy", "embryology", "clinical correlations"],
    "Physiology": ["organ system functions", "homeostasis", "pathophysiology", "regulation pathways"],
    "Biochemistry": ["metabolic pathways", "enzyme kinetics", "molecular biology", "clinical biochemistry"]
}

# Extract PDF text
def extract_text_from_pdfs(pdf_directory):
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            try:
                with open(pdf_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    if pdf_reader.is_encrypted:
                        pdf_reader.decrypt("")
                    full_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                    if full_text.strip():
                        documents.append({"filename": filename, "text": full_text})
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    return documents

# Chunk long text
def chunk_text(doc, chunk_size=500):
    text = doc["text"]
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip() + ". "
        sentence_len = len(sentence)
        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(("".join(current_chunk), doc["filename"]))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len
    if current_chunk:
        chunks.append(("".join(current_chunk), doc["filename"]))
    return chunks

# Build vector DB
@st.cache_resource
def build_vector_db():
    documents = extract_text_from_pdfs(PDF_FILE_PATH)
    if not documents:
        raise ValueError("No readable PDFs found.")

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    all_chunks_with_meta = []
    for doc in documents:
        all_chunks_with_meta.extend(chunk_text(doc))

    all_chunks = [chunk for chunk, meta in all_chunks_with_meta]
    all_metadata = [meta for chunk, meta in all_chunks_with_meta]

    embeddings = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, all_chunks, all_metadata, embedder

# Retrieve relevant chunks
def retrieve_chunks(query, index, chunks, metadata, embedder, k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k)
    return [(chunks[idx], metadata[idx]) for idx in indices[0] if idx != -1]

# Generate one MCQ using Gemini
def generate_mcq(subject, relevant_chunks, previous_mcq_questions):
    model = genai.GenerativeModel(MODEL_ID)
    context = "\n\n".join([f"Source: {filename}\nContent: {chunk}" for chunk, filename in relevant_chunks])
    num_correct = random.randint(1, 5)
    question_focus = random.choice(subject_subtopics.get(subject, ["general"]))

    prompt = f"""
    You are a medical educator. Based *only* on the provided context, create a unique multiple-choice question (MCQ) on {subject} about {question_focus}.
    The MCQ must have 5 options (A, B, C, D, E), with exactly {num_correct} correct answers.
    The question must be different from these: {', '.join(previous_mcq_questions) if previous_mcq_questions else 'None'}.
    The explanation must cite the source filename provided in the context.

    Context:
    ---
    {context}
    ---

    Required Output Format:
    Question: [Your question here]
    A. [Option A]
    B. [Option B]
    C. [Option C]
    D. [Option D]
    E. [Option E]
    Correct Answers: [e.g., A, B, D]
    Explanation: [Your explanation here, citing the source filename.]
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text

        question = re.search(r"Question:\s*(.*)", response_text, re.IGNORECASE)
        options = re.findall(r"([A-E]\.)\s*(.*)", response_text, re.IGNORECASE)
        correct_answers = re.search(r"Correct Answers:\s*(.*)", response_text, re.IGNORECASE)
        explanation = re.search(r"Explanation:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)

        if question and options and correct_answers and explanation:
            parsed_q = question.group(1).strip()
            if parsed_q in previous_mcq_questions:
                return None
            previous_mcq_questions.append(parsed_q)
            return {
                "question": parsed_q,
                "options": [f"{opt[0]} {opt[1].strip()}" for opt in options],
                "correct_answers": [ans.strip() for ans in correct_answers.group(1).split(",")],
                "explanation": explanation.group(1).strip()
            }

    except Exception as e:
        print(f"Error generating MCQ: {e}")
    return None

# Generate multiple MCQs
def generate_questions(subject, count, index, chunks, metadata, embedder):
    results = []
    previous_questions = []
    for _ in range(count):
        subtopic = random.choice(subject_subtopics.get(subject, ["general"]))
        query = f"A first-year medical school question about {subject} focusing on {subtopic}"
        relevant_chunks = retrieve_chunks(query, index, chunks, metadata, embedder)
        if not relevant_chunks:
            continue
        mcq = generate_mcq(subject, relevant_chunks, previous_questions)
        if mcq:
            results.append(mcq)
        time.sleep(1)
    return results

# === Streamlit UI ===
st.title("🧠 Medical MCQ Generator with RAG + Gemini")

with st.sidebar:
    st.header("Settings")
    subject = st.selectbox("Choose a subject", list(subject_subtopics.keys()))
    num_questions = st.slider("Number of Questions", 1, 5, 3)

if st.button("Generate MCQs"):
    with st.spinner("Generating questions..."):
        try:
            index, chunks, metadata, embedder = build_vector_db()
            questions = generate_questions(subject, num_questions, index, chunks, metadata, embedder)
            if not questions:
                st.warning("No questions generated. Try again or check logs.")
            else:
                for i, q in enumerate(questions, 1):
                    st.markdown(f"### Question {i}: {q['question']}")
                    for opt in q['options']:
                        st.markdown(f"- {opt}")
                    st.markdown(f"**Correct Answers**: {', '.join(q['correct_answers'])}")
                    st.markdown(f"**Explanation**: {q['explanation']}")
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error: {e}")
