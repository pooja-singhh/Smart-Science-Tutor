# filename: judge.py (Upgraded with BERTScore)

import pandas as pd
import os
import sys
import subprocess
import ast
import re

# --- Dependency Installation ---
try:
    from rouge_score import rouge_scorer
    import nltk
    from nltk.translate.meteor_score import meteor_score
    from bert_score import score as bert_scorer
    import torch
except ImportError:
    print("Installing required NLP libraries: rouge-score, nltk, bert-score, torch...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rouge-score", "nltk", "bert-score", "torch", "-q"], check=True)
    from rouge_score import rouge_scorer
    import nltk
    from nltk.translate.meteor_score import meteor_score
    from bert_score import score as bert_scorer
    import torch

try:
    from groq import Groq
except ImportError:
    print("Installing Groq client...")
    subprocess.run([sys.executable, "-m", "pip", "install", "groq", "-q"], check=True)
    from groq import Groq


# --- Helper Functions ---

def is_encoded_string(s):
    """
    Uses a regular expression to check if a string is composed of base64-like
    characters and is long enough to be encoded data.
    """
    if not isinstance(s, str) or len(s) < 200:
        return False
    base64_pattern = re.compile(r'^[A-Za-z0-9+/=]+$')
    return bool(base64_pattern.match(s))

def calculate_new_metrics(reference, hypothesis, hallucination_rate):
    """
    Calculates ROUGE, METEOR, BERTScore, IHR, and a final weighted score.
    """
    # --- ROUGE Calculation (ROUGE-L F1) ---
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(reference, hypothesis)
    rouge_l_f1 = rouge_scores['rougeL'].fmeasure

    # --- METEOR Calculation ---
    meteor = meteor_score([reference.split()], hypothesis.split())

    # --- BERTScore Calculation ---
    # Note: BERTScore is computationally more intensive than other metrics.
    # It returns Precision, Recall, and F1 score as tensors.
    P, R, F1 = bert_scorer([hypothesis], [reference], lang='en', verbose=False)
    bert_score_f1 = F1.mean().item()

    # --- Inverse Hallucination Rate ---
    ihr = 1 - hallucination_rate

    # --- New Weights for Final Score ---
    w_ihr = 0.50       # Faithfulness is still the most important.
    w_bert = 0.30      # BERTScore is the primary similarity metric.
    w_meteor = 0.10    # Meteor is still useful for structure.
    w_rouge = 0.10     # ROUGE provides a basic lexical check.

    # --- Final Weighted Score ---
    final_score = (w_ihr * ihr) + (w_bert * bert_score_f1) + \
                  (w_meteor * meteor) + (w_rouge * rouge_l_f1)

    return {
        "ROUGE_L_F1": rouge_l_f1,
        "METEOR": meteor,
        "BERTScore_F1": bert_score_f1,
        "Inverse_Hallucination_Rate": ihr,
        "Final_Score": final_score
    }

def get_faithfulness_score(client, context_text, answer):
    """
    Uses an LLM to rate the faithfulness of an answer to its context.
    Returns a score between 0 and 1.
    """
    faithfulness_prompt = f"""
Rate the faithfulness of the answer to the given context on a scale of 0 to 1, where:
- 1 = The answer is completely faithful and supported by the context.
- 0 = The answer contradicts or is not supported by the context.

Context: {context_text}
Answer: {answer}

Respond with only a single number between 0 and 1 (e.g., 0.8):
"""
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": faithfulness_prompt}],
            temperature=0.0,
            max_tokens=20
        )
        response_text = response.choices[0].message.content.strip()
        match = re.search(r'(\d*\.?\d+)', response_text)
        if match:
            score = float(match.group(0))
            return max(0.0, min(1.0, score))
        else:
            print(f"⚠️  LLM Faithfulness Warning: No number found in response: '{response_text}'. Defaulting to 0.5.")
            return 0.5
    except Exception as e:
        print(f"⚠️  LLM Faithfulness Error: {e}. Defaulting to 0.5.")
        return 0.5

# --- Main Evaluation Logic ---

def judge_the_results(api_key):
    """
    Evaluates results using a combination of LLM-judged faithfulness and
    NLP metrics (ROUGE, METEOR, BERTScore).
    """
    results_path = "evaluation_results.csv"
    if not os.path.exists(results_path):
        print(f"❌ Error: Evaluation results file not found at {results_path}"); return

    print(f"\n🔎 Loading results from {results_path}...")
    try:
        results_df = pd.read_csv(results_path)
        if 'ground_truth' not in results_df.columns:
            print("❌ Error: The CSV must have a 'ground_truth' column.")
            return

        if 'contexts' in results_df.columns:
            results_df['contexts'] = results_df['contexts'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [x]
            )
        if results_df.empty:
            print("❌ The results file is empty."); return
    except Exception as e:
        print(f"❌ Error loading results file: {e}"); return

    print("✅ Results loaded successfully.")
    print("\n🧼 Cleaning context data...")
    results_df['contexts'] = results_df['contexts'].apply(
        lambda ctx_list: [item for item in ctx_list if not is_encoded_string(item)]
    )
    print("✅ Contexts cleaned.")

    client = Groq(api_key=api_key)
    print(f"\n⚖️  Running new evaluation for {len(results_df)} responses...")
    
    all_scores = []
    for idx, row in results_df.iterrows():
        reference = str(row.get('ground_truth', ''))
        hypothesis = str(row.get('answer', ''))
        contexts = row.get('contexts', [])
        context_text = ' '.join([str(item) for item in contexts]) if contexts else ''
        
        faithfulness_score = get_faithfulness_score(client, context_text, hypothesis)
        hallucination_rate = 1.0 - faithfulness_score
        
        if not reference:
            print(f"⚠️  Warning: Row {idx} has empty 'ground_truth'. Scoring with 0 for similarity.")
            metrics = {
                "ROUGE_L_F1": 0.0, "METEOR": 0.0, "BERTScore_F1": 0.0,
                "Inverse_Hallucination_Rate": 1.0 - hallucination_rate,
                "Final_Score": (0.50 * (1.0 - hallucination_rate))
            }
        else:
             metrics = calculate_new_metrics(reference, hypothesis, hallucination_rate)
        
        all_scores.append(metrics)
            
        if (idx + 1) % 5 == 0:
            print(f"✅ Processed {idx + 1}/{len(results_df)} responses")

    scores_df = pd.DataFrame(all_scores)
    final_df = pd.concat([results_df, scores_df], axis=1)
    
    avg_rouge = final_df['ROUGE_L_F1'].mean()
    avg_meteor = final_df['METEOR'].mean()
    avg_bert = final_df['BERTScore_F1'].mean()
    avg_ihr = final_df['Inverse_Hallucination_Rate'].mean()
    avg_final_score = final_df['Final_Score'].mean()

    output_filename = 'detailed_evaluation_results.csv'
    final_df.to_csv(output_filename, index=False)
    
    print("\n🎉🎉🎉 BENCHMARK COMPLETE! 🎉🎉🎉")
    print(f"\n--- Final Scores ---")
    print(f"Average ROUGE-L F1: {avg_rouge:.3f}")
    print(f"Average METEOR: {avg_meteor:.3f}")
    print(f"Average BERTScore F1: {avg_bert:.3f}")
    print(f"Average Inverse Hallucination Rate: {avg_ihr:.3f}")
    print(f"---")
    print(f"Final Weighted Score: {avg_final_score:.3f}")
    
    print(f"\n📄 Detailed results saved to: {output_filename}")
    
    print(f"\n--- Sample Results ---")
    sample_cols = ['question', 'answer', 'Final_Score', 'BERTScore_F1', 'Inverse_Hallucination_Rate']
    print(final_df[sample_cols].head(3).to_string(index=False))

if __name__ == "__main__":
    print("Checking for NLTK data...")
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
        print("✅ NLTK data found.")
    except LookupError:
        print("⬇️  Downloading NLTK data (for METEOR score)...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✅ NLTK data downloaded.")
        
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = input("Please enter your Groq API key: ")
    
    if not api_key:
        print("❌ No API key provided. Aborting evaluation.")
    else:
        if not os.path.exists("evaluation_results.csv"):
            print("❌ 'evaluation_results.csv' not found. Please run your data generation script first.")
        else:
            judge_the_results(api_key)