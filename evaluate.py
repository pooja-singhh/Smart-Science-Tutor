# filename: evaluate.py (The Final, Definitive Version)
import pandas as pd
import os
from tqdm import tqdm
# --- 1. Import all necessary functions and the 'app' module itself ---
from app import (
    setup_application,
    process_hybrid_pdfs,
    get_document_summary_context,
    get_hybrid_enhanced_answer
)
import app

def run_evaluation_questions():
    """
    Runs the RAG system on the test dataset against the pre-loaded knowledge base.
    """
    # --- 1. Load the test dataset ---
    dataset_path = "test_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Test dataset not found at {dataset_path}")
        return
    
    test_df = pd.read_csv(dataset_path)
    questions = test_df["question"].tolist()
    ground_truths = test_df["ground_truth"].tolist()
    
    # --- 2. Run the RAG pipeline for each question ---
    results_for_ragas = []
    print(f"\n🚀 Running evaluation on {len(questions)} questions...")
    
    for question, ground_truth in tqdm(zip(questions, ground_truths), total=len(questions), desc="Evaluating Questions"):
        # Note: We remove the target_pdf parameter because we are querying the entire knowledge base now.
        answer_text, retrieved_images, context_text = get_hybrid_enhanced_answer(
            question,
            []
        )
        
        # Fix: Handle both list and string context_text
        processed_context = ""
        context_list = []
        
        if context_text:
            if isinstance(context_text, list):
                # If it's a list, join non-empty strings and keep the list for contexts
                valid_contexts = [ctx.strip() for ctx in context_text if ctx and str(ctx).strip()]
                if valid_contexts:
                    processed_context = "\n\n".join(valid_contexts)
                    context_list = valid_contexts
            else:
                # If it's a string, use it directly
                if str(context_text).strip():
                    processed_context = str(context_text).strip()
                    context_list = [processed_context]
        
        # Check if we have valid context
        if not processed_context:
            print(f"--> [DEBUG] No context found for question: \"{question}\"")
        
        results_for_ragas.append({
            "question": question,
            "answer": answer_text,
            "contexts": context_list,  # This should now contain the actual context
            "ground_truth": ground_truth
        })
    
    # --- 3. Save the results ---
    results_df = pd.DataFrame(results_for_ragas)
    results_path = "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    
    print(f"\n✅ Evaluation complete. Results saved to: {results_path}")
    print("\n--- Results Preview ---")
    print(results_df.head())

if __name__ == "__main__":
    # --- This is the final, correct setup for multiple PDFs ---
    print("🚀 Setting up the main application environment...")
    setup_application()
    print("✅ Environment is ready.")
    
    # 1. Define the folder containing your PDFs
    pdf_folder = "pdfs used 2"
    if not os.path.isdir(pdf_folder):
        print(f"❌ Error: Folder not found at '{pdf_folder}'")
    else:
        # 2. Create a list of all PDF file paths in the folder
        pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_paths:
            print(f"⚠️ No PDF files found in '{pdf_folder}'.")
        else:
            print(f"📚 Found {len(pdf_paths)} PDFs to process.")
            
            # 3. Check if the knowledge base needs to be built
            if app.faiss_index is None:
                print("⚠️ Knowledge base not pre-loaded. Processing all PDFs now...")
                # Process the entire list of PDFs to build the knowledge base
                process_hybrid_pdfs(pdf_paths)
                print("✅ Knowledge base built successfully.")
            else:
                print("✅ Knowledge base already loaded.")
            
            # 4. Run the evaluation against the complete knowledge base
            run_evaluation_questions()