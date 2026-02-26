# Tweet Classification Backend

Simple Express.js backend for storing and managing tweets for classification.

## Features

- MongoDB integration for tweet storage
- RESTful API endpoints for tweet CRUD operations
- Classification result storage
- No external dependencies (no Twitter API)

## Setup

1. Install dependencies:
```bash
npm install
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your MongoDB URI
```

3. Start server:
```bash
npm start
# or for development with auto-reload:
npm run dev
```

## API Endpoints

### Health Check
- `GET /api/health` - Check server status

### Tweets
- `POST /api/tweets/create` - Create a new tweet
- `GET /api/tweets` - Get all tweets (with filters)
- `GET /api/tweets/:id` - Get single tweet by ID
- `PUT /api/tweets/:id/classify` - Update tweet with classification results
- `DELETE /api/tweets/:id` - Delete a tweet
- `GET /api/tweets/stats/summary` - Get statistics

## Query Parameters (GET /api/tweets)

- `page` - Page number (default: 1)
- `limit` - Items per page (default: 50)
- `labelId` - Filter by classification label ID
- `author` - Filter by author
- `source` - Filter by source (e.g., 'manual')
- `sortBy` - Sort field (default: 'createdAt')
- `sortOrder` - Sort order ('asc' or 'desc', default: 'desc')

## Example Requests

### Create Tweet
```bash
POST /api/tweets/create
{
  "text": "Flooding in downtown area, need help",
  "author": "user123",
  "authorName": "John Doe"
}
```

### Get Tweets
```bash
GET /api/tweets?page=1&limit=20&labelId=0
```

### Update Classification
```bash
PUT /api/tweets/{tweetId}/classify
{
  "classification": {
    "predictedLabelId": 0,
    "predictedLabel": "affected_individuals",
    "confidenceScores": [0.9, 0.05, 0.02, 0.02, 0.01]
  },
  "explanation": [
    {"token": "flooding", "score": 0.95},
    {"token": "help", "score": 0.87}
  ],
  "actionableInfo": {
    "locations": ["downtown"],
    "needs": ["help"]
  }
}
```

## MongoDB Schema

Tweets are stored with the following structure:
- Basic tweet info (text, author, timestamps)
- Classification results (label, confidence scores)
- XAI explanations (token scores)
- Actionable information (locations, people counts, needs, etc.)

## Environment Variables

- `PORT` - Server port (default: 5000)
- `MONGODB_URI` - MongoDB connection string
- `NODE_ENV` - Environment (development/production)
