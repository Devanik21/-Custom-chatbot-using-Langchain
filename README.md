# BrainLox Technical Courses Chatbot

A chatbot built with Langchain and Flask that provides information about technical courses from BrainLox.

## Features

- Extracts course data from [BrainLox Technical Courses](https://brainlox.com/courses/category/technical)
- Creates text embeddings and stores them in a FAISS vector database
- Provides a RESTful API with Flask for conversational interactions
- Retrieves relevant course information based on user queries

## Requirements

- Python 3.8+
- Google API key (with Gemini access)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brainlox-chatbot.git
cd brainlox-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Then edit the `.env` file to add your Google API key for Gemini access.

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. The API will be available at `http://127.0.0.1:5000/api/chat`

3. Send POST requests with JSON in the following format:
```json
{
  "query": "What Python courses are available?",
  "chat_history": [] 
}
```

4. For subsequent messages, include the chat history:
```json
{
  "query": "What is the cost of these courses?",
  "chat_history": [
    ["What Python courses are available?", "BrainLox offers several Python courses including..."]
  ]
}
```

## API Endpoints

- `POST /api/chat`: Main endpoint for chatbot interactions
- `GET /health`: Health check endpoint

## How It Works

1. **Data Extraction**: Uses Langchain's WebBaseLoader to scrape course information from BrainLox.
2. **Text Processing**: Splits content into manageable chunks using RecursiveCharacterTextSplitter.
3. **Vector Storage**: Creates embeddings with Google's Gemini and stores them in a FAISS vector database.
4. **Retrieval Chain**: Uses a ConversationalRetrievalChain to generate contextually relevant answers.

## License

MIT
