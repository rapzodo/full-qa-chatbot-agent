# Full QA Chatbot Agent

This project is a Question Answering (QA) chatbot agent that leverages Retrieval-Augmented Generation (RAG) to answer user queries based on provided documents. It is built using Python and Streamlit for the user interface.

## Features
- Interactive chat UI for asking questions
- RAG-based response generation
- Contextual document retrieval
- Easy integration with custom document files

## Project Structure
```
full-qa-chatbot-agent/
├── requirements.txt
├── files/                  # Directory for source documents (PDFs, etc.)
├── src/
│   └── chatbot/
│       ├── app.py          # Main Streamlit app
│       └── util/
│           └── api_calls.py
└── ...
```

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd full-qa-chatbot-agent
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Start the Streamlit app:
```bash
streamlit run src/chatbot/app.py
```

### Adding Documents
Place your PDF or other supported files in the `files/` directory. The chatbot will use these documents as context for answering questions.

## Usage
- Open the app in your browser (Streamlit will provide a local URL).
- Enter your question in the chat box and submit.
- The chatbot will respond with an answer and show relevant context from your documents.

## License
MIT License

## Author
Danilo De Castro

