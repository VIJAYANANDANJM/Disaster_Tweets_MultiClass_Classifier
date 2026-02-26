const mongoose = require('mongoose');

const tweetSchema = new mongoose.Schema({
  tweetId: { type: String, required: true, unique: true, index: true },
  text: { type: String, required: true },
  author: { type: String, required: true },
  authorName: { type: String, required: true },
  authorId: { type: String, required: true },
  createdAt: { type: Date, required: true, index: true },
  retweetCount: { type: Number, default: 0 },
  favoriteCount: { type: Number, default: 0 },
  url: { type: String },
  
  // UPDATED: Multi-state status for HITL
  status: {
    type: String,
    enum: ['verified', 'unverified', 'human_verified'],
    default: 'unverified',
    index: true
  },

  classification: {
    predictedLabelId: { type: Number },
    predictedLabel: { type: String },
    confidenceScores: { type: [Number] },
    classifiedAt: { type: Date }
  },
  explanation: {
    type: [{ token: String, score: Number }]
  },
  actionableInfo: {
    locations: [String],
    peopleCount: [{ count: Number, status: String }],
    needs: [String],
    damageType: [String],
    timeMentions: [String]
  },
  fetchedAt: { type: Date, default: Date.now },
  source: { type: String, default: 'manual' }
}, {
  timestamps: true
});

tweetSchema.index({ createdAt: -1 });
tweetSchema.index({ status: 1 }); 
module.exports = mongoose.model('Tweet', tweetSchema);