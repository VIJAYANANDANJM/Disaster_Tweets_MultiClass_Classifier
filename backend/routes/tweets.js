const express = require('express');
const router = express.Router();
const Tweet = require('../models/Tweet');

// Create a new tweet (for manual input from dashboard)
router.post('/create', async (req, res) => {
  try {
    const { text, author, authorName, tweetId, createdAt, retweetCount, favoriteCount } = req.body;

    if (!text || !author) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: text and author'
      });
    }

    // Generate tweetId if not provided
    const finalTweetId = tweetId || `manual_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Check if tweet already exists
    const existingTweet = await Tweet.findOne({ 
      $or: [
        { tweetId: finalTweetId },
        { text: text, author: author, createdAt: createdAt ? new Date(createdAt) : { $gte: new Date(Date.now() - 60000) } }
      ]
    });

    if (existingTweet) {
      return res.json({
        success: true,
        message: 'Tweet already exists',
        tweet: existingTweet
      });
    }

    // Create new tweet
    const tweetDoc = new Tweet({
      tweetId: finalTweetId,
      text: text,
      author: author,
      authorName: authorName || author,
      authorId: `manual_${author}`,
      createdAt: createdAt ? new Date(createdAt) : new Date(),
      retweetCount: retweetCount || 0,
      favoriteCount: favoriteCount || 0,
      source: 'manual'
    });

    const savedTweet = await tweetDoc.save();

    res.json({
      success: true,
      message: 'Tweet created successfully',
      tweet: savedTweet
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get all tweets from database
router.get('/', async (req, res) => {
  try {
    const {
      page = 1,
      limit = 50,
      labelId,
      author,
      source,
      sortBy = 'createdAt',
      sortOrder = 'desc'
    } = req.query;

    const query = {};
    
    if (labelId !== undefined && labelId !== null && labelId !== '') {
      query['classification.predictedLabelId'] = parseInt(labelId);
    }
    
    if (author) {
      query.author = author;
    }

    if (source) {
      query.source = source;
    }

    const skip = (parseInt(page) - 1) * parseInt(limit);
    const sortOptions = {};
    sortOptions[sortBy] = sortOrder === 'desc' ? -1 : 1;

    const tweets = await Tweet.find(query)
      .sort(sortOptions)
      .skip(skip)
      .limit(parseInt(limit));

    const total = await Tweet.countDocuments(query);

    res.json({
      success: true,
      tweets,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / parseInt(limit))
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get single tweet by ID
router.get('/:id', async (req, res) => {
  try {
    const tweet = await Tweet.findOne({ tweetId: req.params.id });
    
    if (!tweet) {
      return res.status(404).json({
        success: false,
        error: 'Tweet not found'
      });
    }

    res.json({
      success: true,
      tweet
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Update tweet with classification results (called by desktop dashboard after local classification)
router.put('/:id/classify', async (req, res) => {
  try {
    const tweet = await Tweet.findOne({ tweetId: req.params.id });
    
    if (!tweet) {
      return res.status(404).json({
        success: false,
        error: 'Tweet not found'
      });
    }

    // Update with classification data from desktop app
    const { classification, explanation, actionableInfo } = req.body;
    
    if (classification) {
      tweet.classification = {
        predictedLabelId: classification.predictedLabelId,
        predictedLabel: classification.predictedLabel,
        confidenceScores: classification.confidenceScores,
        classifiedAt: classification.classifiedAt ? new Date(classification.classifiedAt) : new Date()
      };
    }
    
    if (explanation) {
      tweet.explanation = explanation;
    }
    
    if (actionableInfo) {
      tweet.actionableInfo = actionableInfo;
    }

    await tweet.save();

    res.json({
      success: true,
      message: 'Classification saved successfully',
      tweet
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Delete a tweet
router.delete('/:id', async (req, res) => {
  try {
    const tweet = await Tweet.findOneAndDelete({ tweetId: req.params.id });
    
    if (!tweet) {
      return res.status(404).json({
        success: false,
        error: 'Tweet not found'
      });
    }

    res.json({
      success: true,
      message: 'Tweet deleted successfully'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get statistics
router.get('/stats/summary', async (req, res) => {
  try {
    const totalTweets = await Tweet.countDocuments();
    const classifiedTweets = await Tweet.countDocuments({ 'classification.predictedLabelId': { $exists: true } });
    
    const labelCounts = await Tweet.aggregate([
      { $match: { 'classification.predictedLabelId': { $exists: true } } },
      { $group: { _id: '$classification.predictedLabelId', count: { $sum: 1 } } }
    ]);

    res.json({
      success: true,
      stats: {
        totalTweets,
        classifiedTweets,
        unclassifiedTweets: totalTweets - classifiedTweets,
        labelCounts: labelCounts.reduce((acc, item) => {
          acc[item._id] = item.count;
          return acc;
        }, {})
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;
