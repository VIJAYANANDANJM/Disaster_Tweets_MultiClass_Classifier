# CIPPROJECT - Disaster Tweet Classification System

A comprehensive disaster-related tweet classification system with a fine-tuned transformer model, explainable AI (XAI), and actionable information extraction. The system includes a MERN stack web application and a Python desktop dashboard for local model inference.

## 🏗️ Architecture

The system consists of three main components:

1. **Backend (Node.js/Express/MongoDB)** - Stores tweets in database
2. **Frontend (React)** - Web interface for viewing tweets (optional)
3. **Desktop Dashboard (Python)** - **Runs the AI model locally** for classification with XAI visualization

**Key Features**: 
- The AI model **only runs locally** in the desktop dashboard - ensuring privacy and allowing offline operation
- **No Twitter API needed** in dashboard - works with database tweets only
- **Manual tweet input** - Enter tweets directly for classification
- **Database integration** - Load and classify tweets from MongoDB

## 📁 Project Structure

```
CIPPROJECT/
├── backend/                    # Node.js/Express Backend
│   ├── models/
│   │   └── Tweet.js           # MongoDB schema
│   ├── routes/
│   │   ├── auth.js            # Authentication routes
│   │   └── tweets.js          # Tweet CRUD routes
│   ├── services/
│   │   └── twitterService.js  # Twitter API wrapper (for web frontend)
│   ├── server.js              # Express server
│   └── package.json
│
├── frontend/                   # React Frontend (Optional)
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   └── ...
│   └── package.json
│
├── Dashboard/                  # Python Desktop Dashboard ⭐ MODEL RUNS HERE
│   ├── dashboard.py           # Main dashboard UI
│   ├── model_inference.py     # Model wrapper (loads DeLTran15)
│   ├── api_client.py          # Backend API client
│   ├── token_highlighter.py   # Token highlighting logic
│   └── config.py
│
├── Trained_Model/             # Your trained model
│   ├── deltran15_minilm_fp32.pt  # Model weights
│   ├── Model.py               # Model architecture
│   ├── Explainable_AI.py      # XAI implementation
│   ├── Actionable_Info.py     # Actionable info extraction
│   └── Model_Tokenizer/       # Tokenizer files
│
├── Data_Set/                  # Training data
│   ├── Unprocessed_Data_Sets/ # Raw disaster datasets
│   ├── Processed_Data_Set/    # Preprocessed CSVs
│   └── Data_Preprocessing/   # Data prep scripts
│
├── Model_Build/               # Model training scripts
│   └── Build.py
│
├── run_dashboard.py           # Entry point for desktop dashboard
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** (v16+) and npm
- **Python** 3.8+
- **MongoDB** (local or MongoDB Atlas)

### 1. Backend Setup

```bash
cd backend
npm install
cp .env.example .env
# Edit .env with your MongoDB URI
npm start
```

Backend runs on `http://localhost:5000`

### 2. Desktop Dashboard Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model (optional but recommended)
python -m spacy download en_core_web_sm

# Run dashboard
python run_dashboard.py
```

### 3. Frontend Setup (Optional)

```bash
cd frontend
npm install
npm start
```

Frontend runs on `http://localhost:3000`

## 🎯 Classification Categories

The model classifies tweets into 5 categories:

1. **Affected Individuals** 🔴 - People affected by disasters
2. **Infrastructure Damage** 🟠 - Damaged infrastructure
3. **Not Humanitarian** ⚪ - Non-relevant tweets
4. **Other Information** 🔵 - Other relevant information
5. **Rescue/Donation** 🟢 - Rescue efforts or donations

## 💻 Usage

### Desktop Dashboard (Primary Tool)

1. **Start backend**: `cd backend && npm start`
2. **Run dashboard**: `python run_dashboard.py`
3. **Dashboard connects** to backend server automatically
4. **Two ways to get tweets**:
   - **Manual Input**: Enter tweets directly in the dashboard
   - **From Database**: Load tweets already stored in MongoDB
5. **Tweets are automatically classified locally** using your model
6. **View results** with:
   - Color-coded category labels
   - Token-level highlighting (XAI)
   - Actionable information extraction
   - Filter by category

### Web Frontend (Optional)

1. **Start backend**: `cd backend && npm start`
2. **Start frontend**: `cd frontend && npm start`
3. **Login** with Twitter API credentials (for fetching tweets)
4. **Fetch tweets** from Twitter (stored in MongoDB)
5. **View tweets** in web interface (classification happens in desktop app)

## 📊 Data Flow

```
1. Get Tweets (Two Methods):
   a) Manual Input: Dashboard → User enters tweet → Saved to MongoDB
   b) From Database: Dashboard → Backend → MongoDB → Load tweets

2. Classification (Desktop Dashboard Only):
   Dashboard → Loads DeLTran15 model locally → Classifies tweets
   → Generates XAI explanations → Extracts actionable info
   → Saves classification results to MongoDB

3. View Results:
   Dashboard → Backend → MongoDB → Display classified tweets
   → Filter by category → View details with token highlighting
```

## 🔬 Model Details

### Architecture

- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Custom Head**: Linear classifier on `[CLS]` embedding with dropout
- **Weights**: `Trained_Model/deltran15_minilm_fp32.pt`
- **Classes**: 5 disaster-related categories

### Explainable AI (XAI)

- Token-level importance scores using gradient-based methods
- Visual highlighting with color gradients (white → red)
- Shows which tokens are most influential for classification

### Actionable Information Extraction

For actionable categories, the system extracts:
- 📍 **Locations** - Geographic locations (via spaCy NER)
- 👥 **People Counts** - Number of affected people
- 🆘 **Needs** - Required resources (food, water, medicine, etc.)
- 💥 **Damage Types** - Types of damage mentioned
- ⏰ **Time Mentions** - Temporal information

## 🔒 Privacy & Security

- ✅ **Model runs locally** - Never leaves your machine
- ✅ **No cloud classification** - All inference happens on your computer
- ✅ **Privacy-first** - Your data stays private
- ✅ **Offline capable** - Can work without internet (after fetching tweets)
- ✅ **No Twitter API in dashboard** - Works with database only

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup instructions
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Architecture overview
- **[Dashboard/README.md](Dashboard/README.md)** - Dashboard documentation
- **[README_MERN.md](README_MERN.md)** - MERN stack details

## 🛠️ Development

### Training the Model

```bash
cd Model_Build
python Build.py
```

### Data Preprocessing

```bash
cd Data_Set/Data_Preprocessing
python Data_Extraction.py
python Data_Cleaning.py
```

### Running CLI Classifier

```bash
python Trained_Model/Main.py
```

## 📦 Dependencies

### Python (Desktop Dashboard)
- `torch` - Deep learning framework
- `transformers` - Hugging Face transformers
- `customtkinter` - Modern UI framework
- `spacy` - NLP for actionable info extraction
- `requests` - HTTP client for backend API

### Node.js (Backend)
- `express` - Web framework
- `mongoose` - MongoDB ODM
- `twitter-api-v2` - Twitter API wrapper (for web frontend)
- `cors` - CORS middleware

### React (Frontend)
- `react` - UI framework
- `@mui/material` - Component library
- `axios` - HTTP client

## 🐛 Troubleshooting

- **Model not loading**: Ensure `Trained_Model/deltran15_minilm_fp32.pt` exists
- **Backend errors**: Check MongoDB is running and `.env` is configured
- **Dashboard connection errors**: Ensure backend is running on port 5000
- **Import errors**: Install all dependencies from `requirements.txt`

## 📝 Notes

- The model weights (`deltran15_minilm_fp32.pt`) are large files
- Classification happens **only** in the desktop dashboard
- Backend stores raw tweets and receives classification results
- Dashboard works with database tweets only (no Twitter API needed)
- Frontend is optional - dashboard can work standalone

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Fine-tuned DeLTran15 model for disaster classification
- Twitter API for tweet data (via web frontend)
- Hugging Face for transformer models
