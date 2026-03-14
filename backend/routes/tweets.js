const express = require('express');
const router = express.Router();
const Tweet = require('../models/Tweet');

// Create a new tweet (for manual input from dashboard)
router.post('/create', async (req, res) => {
  try {
    const {
      text, author, authorName, authorId, tweetId, createdAt,
      retweetCount, favoriteCount, status, source,
      // Geospatial fields
      placeTag, placeCountry, userProfileLocation, geoCoordinates
    } = req.body;

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

    // Resolve location using 3-tier priority: placeTag > userProfileLocation > (text extraction later)
    let resolvedLocation = '';
    let locationSource = '';
    if (placeTag) {
      resolvedLocation = placeTag;
      locationSource = 'place_tag';
    } else if (userProfileLocation) {
      resolvedLocation = userProfileLocation;
      locationSource = 'user_profile';
    }

    // Create new tweet
    const tweetDoc = new Tweet({
      tweetId: finalTweetId,
      text: text,
      author: author,
      authorName: authorName || author,
      authorId: authorId || `manual_${author}`,
      createdAt: createdAt ? new Date(createdAt) : new Date(),
      retweetCount: retweetCount || 0,
      favoriteCount: favoriteCount || 0,
      status: status || 'unverified',
      source: source || 'manual',
      // Geospatial fields
      location: resolvedLocation,
      locationSource: locationSource,
      placeTag: placeTag || '',
      placeCountry: placeCountry || '',
      userProfileLocation: userProfileLocation || '',
      geoCoordinates: geoCoordinates || undefined
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
      status,
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

    if (status) {
      query.status = status;
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

// ── Geospatial Aggregation Endpoint ──────────────────────────────
router.get('/geo-aggregate', async (req, res) => {
  try {
    const minClusterSize = parseInt(req.query.minClusterSize) || 5;

    // MongoDB aggregation: group tweets by location
    const clusters = await Tweet.aggregate([
      // Only include tweets that have a resolved location
      { $match: { location: { $ne: '' }, location: { $exists: true } } },
      // Group by location
      {
        $group: {
          _id: '$location',
          totalTweets: { $sum: 1 },
          uniqueAuthors: { $addToSet: '$author' },
          tweets: { $push: '$$ROOT' },
          locationSource: { $first: '$locationSource' },
          placeCountry: { $first: '$placeCountry' },
          earliestTweet: { $min: '$createdAt' },
          latestTweet: { $max: '$createdAt' },
          // 5-class label distribution (for classified tweets)
          label0Count: {
            $sum: { $cond: [{ $eq: ['$classification.predictedLabelId', 0] }, 1, 0] }
          },
          label1Count: {
            $sum: { $cond: [{ $eq: ['$classification.predictedLabelId', 1] }, 1, 0] }
          },
          label2Count: {
            $sum: { $cond: [{ $eq: ['$classification.predictedLabelId', 2] }, 1, 0] }
          },
          label3Count: {
            $sum: { $cond: [{ $eq: ['$classification.predictedLabelId', 3] }, 1, 0] }
          },
          label4Count: {
            $sum: { $cond: [{ $eq: ['$classification.predictedLabelId', 4] }, 1, 0] }
          },
          classifiedCount: {
            $sum: { $cond: [{ $gt: ['$classification.predictedLabelId', null] }, 1, 0] }
          }
        }
      },
      // Filter by minimum cluster size
      { $match: { totalTweets: { $gte: minClusterSize } } },
      // Sort by tweet count descending
      { $sort: { totalTweets: -1 } },
      // Format output
      {
        $project: {
          _id: 0,
          location: '$_id',
          totalTweets: 1,
          uniqueAuthors: { $size: '$uniqueAuthors' },
          locationSource: 1,
          placeCountry: 1,
          earliestTweet: 1,
          latestTweet: 1,
          timeSpanHours: {
            $divide: [{ $subtract: ['$latestTweet', '$earliestTweet'] }, 3600000]
          },
          classifiedCount: 1,
          labelDistribution: {
            affectedIndividuals: '$label0Count',
            infrastructureDamage: '$label1Count',
            notHumanitarian: '$label2Count',
            otherInformation: '$label3Count',
            rescueDonation: '$label4Count'
          },
          tweets: 1
        }
      }
    ]);

    res.json({
      success: true,
      totalClusters: clusters.length,
      minClusterSize,
      clusters
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

    const { classification, explanation, actionableInfo, status, location, locationSource } = req.body;

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

      // If location was extracted from text and no location was set before, use it
      if (!tweet.location && actionableInfo.locations && actionableInfo.locations.length > 0) {
        tweet.location = actionableInfo.locations[0];
        tweet.locationSource = 'text_extraction';
      }
    }

    // Allow explicit location override
    if (location) {
      tweet.location = location;
      tweet.locationSource = locationSource || 'text_extraction';
    }

    if (status) {
      tweet.status = status;
    }

    await tweet.save();

    res.json({
      success: true,
      message: 'Classification and status updated successfully',
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

module.exports = router;