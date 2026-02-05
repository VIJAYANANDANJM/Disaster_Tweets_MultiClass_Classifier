const mongoose = require('mongoose');

const tweetSchema = new mongoose.Schema({
  tweetId: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  text: {
    type: String,
    required: true
  },
  author: {
    type: String,
    required: true
  },
  authorName: {
    type: String,
    required: true
  },
  authorId: {
    type: String,
    required: true
  },
  createdAt: {
    type: Date,
    required: true,
    index: true
  },
  retweetCount: {
    type: Number,
    default: 0
  },
  favoriteCount: {
    type: Number,
    default: 0
  },
  url: {
    type: String
  },
  // Classification results (added by desktop dashboard)
  classification: {
    predictedLabelId: {
      type: Number
    },
    predictedLabel: {
      type: String
    },
    confidenceScores: {
      type: [Number]
    },
    classifiedAt: {
      type: Date
    }
  },
  // XAI explanation (token scores)
  explanation: {
    type: [{
      token: String,
      score: Number
    }]
  },
  // Actionable information
  actionableInfo: {
    locations: [String],
    peopleCount: [{
      count: Number,
      status: String
    }],
    needs: [String],
    damageType: [String],
    timeMentions: [String]
  },
  // Metadata
  fetchedAt: {
    type: Date,
    default: Date.now
  },
  source: {
    type: String,
    default: 'manual'
  }
}, {
  timestamps: true
});

// Indexes for efficient queries
tweetSchema.index({ createdAt: -1 });
tweetSchema.index({ 'classification.predictedLabelId': 1 });
tweetSchema.index({ author: 1 });
tweetSchema.index({ source: 1 });

module.exports = mongoose.model('Tweet', tweetSchema);
