

import os
import pymupdf as fitz
import pickle
import ssl
import nltk
import base64
import asyncio
from anthropic import AsyncAnthropic # Import the Anthropic async client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time

# --- SSL and NLTK Setup ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading NLTK resources (punkt, stopwords)...")
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
print("NLTK resources downloaded.")

# --- Configuration ---
PDF_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "pdfs")
MD_EXPLANATIONS_PATH = os.path.join(os.path.dirname(__file__), "markdown_explanations")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "hybrid_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Claude API Configuration ---
# Create a single, reusable client instance for efficiency
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY1"))

async def get_explanation_from_image(image_base64: str, prompt: str) -> str:
    """Sends an image to the Claude 4 Vision API and returns the text explanation."""
    try:
        response = await anthropic_client.messages.create(
            # Using Claude 3.5 Sonnet as it's a great balance of speed, cost, and power for this task.
            # model="claude-3-5-sonnet-20241022",
            model="claude-4-sonnet-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        return response.content[0].text
    except Exception as e:
        print(f"An unexpected error occurred calling Claude API: {e}")
        return f"Error processing page with Claude: {e}"

async def process_all_pdfs():
    """
    The main new function to convert all PDF pages to images and generate Markdown explanations.
    """
    print("\n--- Starting Stage 1: PDF to Markdown Explanation Generation ---")
    if not os.path.exists(MD_EXPLANATIONS_PATH):
        os.makedirs(MD_EXPLANATIONS_PATH)

    pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.endswith(".pdf")]
    
    for pdf_file in pdf_files:
        file_path = os.path.join(PDF_FOLDER_PATH, pdf_file)
        print(f"\nProcessing {pdf_file}...")
        
        doc = fitz.open(file_path)
        tasks = []
        
        for i, page in enumerate(doc):
            # Convert page to a PNG image
            pix = page.get_pixmap(dpi=200) # Higher DPI for better quality
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Create the prompt for the LLM
            prompt = """You are an expert data analyst. This image is a page from a document. 
                Analyze any tables, charts, or complex layouts present in the image. For tables:
                1. Process each table row by row, and provide a detailed explanation of the information in each row.
                2. Create a table itself in Markdown format, ensuring that all data is captured accurately.
                3. If there are multiple tables, explain each table separately, row by row.

                For text data:
                1. Summarize the text content comprehensively.
                2. Elaborate on the meaning and context of the text to ensure no information is missed.

                Your goal is to create a detailed and exhaustive explanation of all the information on this page. 
                Format your response in clean Markdown, ensuring proper headings, bullet points, and structure for clarity."""
            
            # Add the async task to a list
            tasks.append(get_explanation_from_image(img_base64, prompt))

        # Run all page processing for the current PDF concurrently
        explanations = await asyncio.gather(*tasks)
        
        # Save the generated explanations to Markdown files
        for i, explanation in enumerate(explanations):
            md_filename = f"{os.path.splitext(pdf_file)[0]}_page_{i+1}.md"
            md_filepath = os.path.join(MD_EXPLANATIONS_PATH, md_filename)
            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(explanation)
            print(f"  -> Saved explanation for page {i+1} to {md_filename}")
            
        doc.close()

def build_hybrid_index_from_markdown():
    """
    Reads the generated Markdown files, chunks them, and builds the hybrid index.
    """
    print("\n--- Starting Stage 2: Building Hybrid Index from Markdown Files ---")
    
    all_docs = []
    md_files = [f for f in os.listdir(MD_EXPLANATIONS_PATH) if f.endswith(".md")]
    if not md_files:
        print("No Markdown explanation files found. Please run Stage 1 first.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for md_file in md_files:
        file_path = os.path.join(MD_EXPLANATIONS_PATH, md_file)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_docs.append({"text": chunk, "metadata": {"source": md_file, "chunk_id": i}})
    
    print(f"Created a total of {len(all_docs)} text chunks from Markdown files.")
    texts = [doc["text"] for doc in all_docs]
    metadatas = [doc["metadata"] for doc in all_docs]

    # Create FAISS and BM25 indices as before
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Creating FAISS index...")
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    
    print("Creating BM25 index...")
    stop_words = set(stopwords.words('english'))
    tokenized_corpus = [[word for word in word_tokenize(doc.lower()) if word.isalpha() and word not in stop_words] for doc in texts]
    bm25_index = BM25Okapi(tokenized_corpus)

    if not os.path.exists(INDEX_PATH):
        os.makedirs(INDEX_PATH)

    vector_store.save_local(os.path.join(INDEX_PATH, "faiss_index"))
    with open(os.path.join(INDEX_PATH, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25_index, f)
    with open(os.path.join(INDEX_PATH, "documents.pkl"), "wb") as f:
        pickle.dump(all_docs, f)

    print("-" * 50)
    print(f"Hybrid index successfully built and saved to '{INDEX_PATH}'")
    print("-" * 50)

async def main():
    start_time = time.time()
    await process_all_pdfs()
    build_hybrid_index_from_markdown()
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())

