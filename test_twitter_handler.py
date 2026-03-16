"""
Tests for Twitter Handler v2 Filters
"""

import unittest
from unittest.mock import MagicMock, patch

from twitter_handler import TwitterHandler, TwitterV2Filters


class TestTwitterV2Filters(unittest.TestCase):
    """Test cases for TwitterV2Filters TypedDict."""
    
    def test_v2_filters_can_be_created(self):
        """Test that TwitterV2Filters can be instantiated."""
        filters: TwitterV2Filters = {
            "has_media": True,
            "has_links": False,
            "exclude_retweets": True,
            "exclude_replies": True,
            "language": "en",
            "min_retweets": 10,
            "max_age_hours": 6
        }
        
        self.assertTrue(filters["has_media"])
        self.assertFalse(filters["has_links"])
        self.assertEqual(filters["min_retweets"], 10)


class TestTwitterHandlerV2QueryBuilder(unittest.TestCase):
    """Test cases for TwitterHandler v2 query building."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "twitter": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "access_token": "test_token",
                "access_token_secret": "test_token_secret",
                "bearer_token": "test_bearer",
                "task": "Test task",
                "search_keywords": ["python", "machine learning"],
                "scan_interval_minutes": 5,
                "auto_reply": False,
                "v2_filters": {}
            }
        }
        self.llm_callback = MagicMock(return_value="Test response")
        
        # Mock tweepy so we don't need real credentials
        with patch.dict('sys.modules', {'tweepy': MagicMock()}):
            self.handler = TwitterHandler(self.config, self.llm_callback)
    
    def test_build_v2_query_basic(self):
        """Test basic query building with keywords."""
        keywords = ["python", "data science"]
        query = self.handler._build_v2_query(keywords, {})
        
        self.assertIn("python", query)
        self.assertIn("data science", query)
        self.assertIn("OR", query)
    
    def test_build_v2_query_with_phrases(self):
        """Test query building with multi-word phrases."""
        keywords = ["python", "machine learning"]
        query = self.handler._build_v2_query(keywords, {})
        
        # Multi-word phrases should be quoted
        self.assertIn('"machine learning"', query)
        self.assertIn("python", query)
    
    def test_build_v2_query_exclude_retweets(self):
        """Test query includes -is:retweet filter."""
        filters: TwitterV2Filters = {"exclude_retweets": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("-is:retweet", query)
    
    def test_build_v2_query_exclude_replies(self):
        """Test query includes -is:reply filter."""
        filters: TwitterV2Filters = {"exclude_replies": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("-is:reply", query)
    
    def test_build_v2_query_exclude_quotes(self):
        """Test query includes -is:quote filter."""
        filters: TwitterV2Filters = {"exclude_quotes": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("-is:quote", query)
    
    def test_build_v2_query_exclude_nullcast(self):
        """Test query includes -is:nullcast filter."""
        filters: TwitterV2Filters = {"exclude_nullcast": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("-is:nullcast", query)
    
    def test_build_v2_query_has_media(self):
        """Test query includes has:media filter."""
        filters: TwitterV2Filters = {"has_media": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("has:media", query)
    
    def test_build_v2_query_has_links(self):
        """Test query includes has:links filter."""
        filters: TwitterV2Filters = {"has_links": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("has:links", query)
    
    def test_build_v2_query_has_hashtags(self):
        """Test query includes has:hashtags filter."""
        filters: TwitterV2Filters = {"has_hashtags": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("has:hashtags", query)
    
    def test_build_v2_query_has_mentions(self):
        """Test query includes has:mentions filter."""
        filters: TwitterV2Filters = {"has_mentions": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("has:mentions", query)
    
    def test_build_v2_query_has_images(self):
        """Test query includes has:images filter."""
        filters: TwitterV2Filters = {"has_images": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("has:images", query)
    
    def test_build_v2_query_has_videos(self):
        """Test query includes has:videos filter."""
        filters: TwitterV2Filters = {"has_videos": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("has:videos", query)
    
    def test_build_v2_query_has_geo(self):
        """Test query includes has:geo filter."""
        filters: TwitterV2Filters = {"has_geo": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("has:geo", query)
    
    def test_build_v2_query_is_verified(self):
        """Test query includes is:verified filter."""
        filters: TwitterV2Filters = {"is_verified": True}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("is:verified", query)
    
    def test_build_v2_query_language(self):
        """Test query includes language filter."""
        filters: TwitterV2Filters = {"language": "de"}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("lang:de", query)
    
    def test_build_v2_query_conversation_id(self):
        """Test query includes conversation_id filter."""
        filters: TwitterV2Filters = {"conversation_id": "12345"}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("conversation_id:12345", query)
    
    def test_build_v2_query_context_entities(self):
        """Test query includes context entity filters."""
        filters: TwitterV2Filters = {"context_entity_ids": ["131.1234", "66.5678"]}
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("context:131.1234", query)
        self.assertIn("context:66.5678", query)
    
    def test_build_v2_query_combined_filters(self):
        """Test query with multiple combined filters."""
        filters: TwitterV2Filters = {
            "exclude_retweets": True,
            "exclude_replies": True,
            "has_media": True,
            "is_verified": True,
            "language": "en"
        }
        query = self.handler._build_v2_query(["test"], filters)
        
        self.assertIn("-is:retweet", query)
        self.assertIn("-is:reply", query)
        self.assertIn("has:media", query)
        self.assertIn("is:verified", query)
        self.assertIn("lang:en", query)


class TestTwitterHandlerEngagementFilter(unittest.TestCase):
    """Test cases for engagement filtering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "twitter": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "access_token": "test_token",
                "access_token_secret": "test_token_secret",
                "bearer_token": "test_bearer",
                "task": "Test task",
                "search_keywords": ["python"],
                "v2_filters": {}
            }
        }
        self.llm_callback = MagicMock(return_value="Test response")
        
        with patch.dict('sys.modules', {'tweepy': MagicMock()}):
            self.handler = TwitterHandler(self.config, self.llm_callback)
    
    def test_filter_by_min_retweets(self):
        """Test filtering by minimum retweets."""
        tweets = [
            {"id": "1", "metrics": {"retweet_count": 5, "like_count": 10, "reply_count": 2}},
            {"id": "2", "metrics": {"retweet_count": 15, "like_count": 20, "reply_count": 3}},
            {"id": "3", "metrics": {"retweet_count": 25, "like_count": 30, "reply_count": 5}},
        ]
        
        filters: TwitterV2Filters = {"min_retweets": 10}
        filtered = self.handler._filter_by_engagement(tweets, filters)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["id"], "2")
        self.assertEqual(filtered[1]["id"], "3")
    
    def test_filter_by_min_likes(self):
        """Test filtering by minimum likes."""
        tweets = [
            {"id": "1", "metrics": {"retweet_count": 5, "like_count": 5, "reply_count": 2}},
            {"id": "2", "metrics": {"retweet_count": 15, "like_count": 25, "reply_count": 3}},
            {"id": "3", "metrics": {"retweet_count": 25, "like_count": 35, "reply_count": 5}},
        ]
        
        filters: TwitterV2Filters = {"min_likes": 20}
        filtered = self.handler._filter_by_engagement(tweets, filters)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["id"], "2")
        self.assertEqual(filtered[1]["id"], "3")
    
    def test_filter_by_min_replies(self):
        """Test filtering by minimum replies."""
        tweets = [
            {"id": "1", "metrics": {"retweet_count": 5, "like_count": 10, "reply_count": 1}},
            {"id": "2", "metrics": {"retweet_count": 15, "like_count": 20, "reply_count": 5}},
            {"id": "3", "metrics": {"retweet_count": 25, "like_count": 30, "reply_count": 10}},
        ]
        
        filters: TwitterV2Filters = {"min_replies": 5}
        filtered = self.handler._filter_by_engagement(tweets, filters)
        
        self.assertEqual(len(filtered), 2)
    
    def test_filter_combined_engagement(self):
        """Test filtering with multiple engagement thresholds."""
        tweets = [
            {"id": "1", "metrics": {"retweet_count": 5, "like_count": 10, "reply_count": 2}},
            {"id": "2", "metrics": {"retweet_count": 15, "like_count": 25, "reply_count": 5}},
            {"id": "3", "metrics": {"retweet_count": 25, "like_count": 35, "reply_count": 8}},
        ]
        
        filters: TwitterV2Filters = {"min_retweets": 10, "min_likes": 20, "min_replies": 4}
        filtered = self.handler._filter_by_engagement(tweets, filters)
        
        self.assertEqual(len(filtered), 2)
    
    def test_no_engagement_filters_returns_all(self):
        """Test that no engagement filters returns all tweets."""
        tweets = [
            {"id": "1", "metrics": {"retweet_count": 5, "like_count": 10, "reply_count": 2}},
            {"id": "2", "metrics": {"retweet_count": 15, "like_count": 25, "reply_count": 5}},
        ]
        
        filtered = self.handler._filter_by_engagement(tweets, {})
        
        self.assertEqual(len(filtered), 2)


class TestTwitterHandlerV2Status(unittest.TestCase):
    """Test cases for Twitter handler status including v2 filters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "twitter": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "access_token": "test_token",
                "access_token_secret": "test_token_secret",
                "bearer_token": "test_bearer",
                "task": "Test task",
                "search_keywords": ["python"],
                "v2_filters": {
                    "has_media": True,
                    "language": "en"
                }
            }
        }
        self.llm_callback = MagicMock(return_value="Test response")
        
        with patch.dict('sys.modules', {'tweepy': MagicMock()}):
            self.handler = TwitterHandler(self.config, self.llm_callback)
    
    def test_status_includes_v2_filters(self):
        """Test that get_status includes v2_filters."""
        status = self.handler.get_status()
        
        self.assertIn("v2_filters", status)
        self.assertEqual(status["v2_filters"]["has_media"], True)
        self.assertEqual(status["v2_filters"]["language"], "en")
    
    def test_get_v2_filters(self):
        """Test get_v2_filters method."""
        filters = self.handler.get_v2_filters()
        
        self.assertEqual(filters["has_media"], True)
        self.assertEqual(filters["language"], "en")
    
    def test_set_v2_filters(self):
        """Test set_v2_filters method."""
        new_filters: TwitterV2Filters = {
            "has_videos": True,
            "min_retweets": 50,
            "language": "de"
        }
        
        result = self.handler.set_v2_filters(new_filters)
        
        self.assertTrue(result["success"])
        self.assertEqual(self.handler.config["v2_filters"]["has_videos"], True)
        self.assertEqual(self.handler.config["v2_filters"]["min_retweets"], 50)
        self.assertEqual(self.handler.config["v2_filters"]["language"], "de")


if __name__ == "__main__":
    unittest.main()
