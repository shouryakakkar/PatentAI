# PatentAI - AI-Powered Patent Search Tool

PatentAI is a modern web application that helps users search and analyze patents using advanced AI technology. It provides both semantic and keyword-based search capabilities, along with features for collecting user feedback to improve search results.

## Features

- **AI-Powered Semantic Search**: Find relevant patents using natural language queries
- **Keyword Search**: Traditional search based on exact word matches
- **Advanced Patent Information**: View detailed patent information including title, abstract, inventors, assignee, and more
- **Feedback Collection**: Help improve the search results by providing relevance feedback
- **Bookmarking**: Save interesting patents for later reference
- **Sharing**: Share patents with others easily

## Tech Stack

### Backend
- Python 3.8+
- FastAPI
- Sentence Transformers
- USPTO API
- SQLite

### Frontend
- React
- Material-UI
- Axios

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PatentAI.git
cd PatentAI
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

### Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn app.main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Project Structure

```
PatentAI/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── model_training.py
│   │   └── patent_search.py
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   └── ...
│   ├── package.json
│   └── README.md
├── .gitignore
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- USPTO for providing the patent data API
- Hugging Face for the sentence-transformers library
- FastAPI and React communities for their excellent documentation 