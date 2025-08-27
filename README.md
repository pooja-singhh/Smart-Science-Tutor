# EdgeLearn AI 🧠

EdgeLearn AI is a smart tutor that lets you have a voice conversation with your PDF documents using a locally-run RAG (Retrieval-Augmented Generation) pipeline.



## ✨ Features

* **PDF Processing**: Extracts text and images from PDFs for semantic understanding.
* **Voice Interaction**: Ask questions and hear responses using offline, local Text-to-Speech.
* **Hybrid Search**: Uses a combination of vector search and knowledge graphs to find the most relevant information.
* **Local First**: All models (LLM, Whisper, TTS) run completely on your local machine.

## 🚀 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/edgelearn.git](https://github.com/your-username/edgelearn.git)
    cd edgelearn
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install Clip from git hub:**
    ```bash
    pip install git+https://github.com/openai/CLIP.git
    ```
3.  **Install Piper TTS**
    ```bash
    git clone https://github.com/rhasspy/piper.git --recursive
    cd piper
    mkdir build
    cd build
    cmake ..
    make
    ```

## 🏃‍♀️ How to Run

1.  **Run the main application:**
    ```bash
    python3 app.py
    ```
    This will start the Gradio web interface. The script will automatically download the necessary AI models on the first run.

2.  **Run the evaluation script:**
    ```bash
    python3 evaluate.py
    python3 judge.py
    ```

## 🛠️ Technologies Used

* **Backend**: Python, Gradio
* **AI Models**: GPT4All, Sentence-Transformers, Whisper, CLIP
* **Databases**: FAISS, SQLite, NetworkX