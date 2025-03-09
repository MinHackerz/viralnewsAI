import os
import logging
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import facebook
import yaml
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import pytz
import hashlib
import imghdr
import time
from PIL import Image
import io
import random

# Suppress SSL warnings
import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Data class to hold news article information"""
    title: str
    url: str
    content: str
    published_at: datetime
    source_name: str
    thumbnail_url: Optional[str] = None

@dataclass
class ProcessedContent:
    """Data class to hold processed content"""
    summary: str
    hashtags: List[str]
    image_path: Optional[str] = None

class APIClient(ABC):
    """Abstract base class for API clients"""
    @abstractmethod
    def validate_credentials(self) -> bool:
        pass

    @abstractmethod
    def handle_rate_limits(self) -> None:
        pass

class NewsDataIOClient(APIClient):
    """Client for handling NewsData.io API interactions"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsdata.io/api/1/latest"

    def validate_credentials(self) -> bool:
        """Validate the API key by making a small request."""
        try:
            response = requests.get(
                f"{self.base_url}?apikey={self.api_key}&country=in&language=en&size=1",
                timeout=30
            )
            data = response.json()
            return data.get("status") == "success"
        except Exception as e:
            logger.error(f"Failed to validate NewsData.io credentials: {str(e)}")
            return False

    def handle_rate_limits(self) -> None:
        """Basic rate limit handling."""
        pass

    def fetch_latest_news(self, country: str, size: int) -> List[Dict]:
        """Fetch the latest news from NewsData.io from top sources."""
        try:
            response = requests.get(
                f"{self.base_url}?apikey={self.api_key}&country={country}&language=en&size={size}&prioritydomain=top",
                timeout=30
            )
            data = response.json()
            if data.get("status") != "success":
                logger.error(f"NewsData.io API error: {data.get('message', 'Unknown error')}")
                time.sleep(60)  # Wait 60 seconds on error (e.g., rate limit)
                return []
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Error fetching news from NewsData.io: {str(e)}")
            return []

class Config:
    """Configuration manager"""
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}")
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        default_config = {
            'content': {
                'max_articles': 5,
                'summary_length': 60,  # Changed to 60 words
                'hashtag_count': 10   # Base count, will rotate between 10-15
            }
        }
        with open(self.config_path, 'w') as file:
            yaml.dump(default_config, file)
        return default_config

class NewsAutomation:
    """Main class for news automation system"""
    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self._initialize_clients()
        self._validate_setup()
        self.posted_articles_file = "posted_articles.txt"
        self.posted_articles = self._load_posted_articles()
        self._ensure_image_dir()

    def _ensure_image_dir(self):
        os.makedirs("images", exist_ok=True)

    def _initialize_clients(self) -> None:
        self.news_api = NewsDataIOClient(os.getenv('NEWSAPI_KEY'))
        self.fb_api = facebook.GraphAPI(access_token=os.getenv('FACEBOOK_ACCESS_TOKEN'))
        self.page_id = "530122440176152"

    def _validate_setup(self) -> None:
        required_env_vars = ['FACEBOOK_ACCESS_TOKEN', 'NEWSAPI_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing environment variables: {missing_vars}")
        if not self.news_api.validate_credentials():
            raise ValueError("Invalid NewsData.io credentials")

    def get_viral_news(self) -> List[NewsArticle]:
        """Fetch the most recent news specific to India from legitimate sources."""
        articles = self.news_api.fetch_latest_news(
            country="in",
            size=self.config.config['content']['max_articles']
        )
        if not articles:
            logger.warning("No recent articles found for India")
            return []

        news_articles = []
        for article in articles:
            try:
                pub_date = datetime.strptime(article["pubDate"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)
            except ValueError as e:
                logger.error(f"Failed to parse pubDate '{article['pubDate']}': {str(e)}. Using current time.")
                pub_date = datetime.now(pytz.UTC)

            news_articles.append(
                NewsArticle(
                    title=article["title"],
                    url=article["link"],
                    content=article.get("description", ""),
                    published_at=pub_date,
                    source_name=article["source_name"],
                    thumbnail_url=article.get("image_url")
                )
            )
        logger.info(f"Found {len(news_articles)} recent articles for India")
        return news_articles

    def process_article(self, article: NewsArticle) -> Optional[ProcessedContent]:
        try:
            summary = self._generate_summary(article.title, article.content)
            hashtags = self._generate_hashtags(article.content)
            image_path = self._save_thumbnail(article.thumbnail_url)
            if not image_path:
                logger.error("Failed to save high-quality thumbnail for the article")
                return None
            return ProcessedContent(summary=summary, hashtags=hashtags, image_path=image_path)
        except Exception as e:
            logger.error(f"Error processing article: {str(e)}")
            return None

    def _generate_summary(self, title: str, content: str) -> str:
        """Craft a 60-word summary using title and description, ensuring it’s human-like."""
        # If content (description) exists and is long enough, summarize it with title context
        if content and len(content.split()) > 20:
            words = content.split()[:50]  # Take enough words to trim to 60 with title
            summary_base = " ".join(words).strip(".,")  # Clean up trailing punctuation
            summary = f"Breaking: {title}. {summary_base} unfolds across India, captivating the nation with its impact."
        else:
            # Fallback to title-only summary, padded to 60 words
            summary = (
                f"Breaking news: {title}. A significant event shakes India, drawing widespread attention. "
                f"Details are emerging as this story develops, keeping the public on edge. Stay tuned."
            )

        # Trim or pad to exactly 60 words
        summary_words = summary.split()
        if len(summary_words) > 60:
            summary = " ".join(summary_words[:60])
        elif len(summary_words) < 60:
            summary += " Follow this unfolding saga as more updates roll in from across the country."
        summary = " ".join(summary.split()[:60])  # Ensure exact 60 words

        logger.info(f"Generated summary: {summary}")
        return summary

    def _generate_hashtags(self, text: str) -> List[str]:
        try:
            hashtag_count = random.randint(10, 15)  # Rotate between 10-15
            # Simple hashtag generation from text
            words = [w.capitalize() for w in text.split() if len(w) > 4 and w.isalpha()]
            hashtags = [f"#{w}" for w in words[:hashtag_count]]
            if len(hashtags) < hashtag_count:
                hashtags.extend(self._fallback_hashtags(text, hashtag_count - len(hashtags)))
            return hashtags[:hashtag_count]
        except Exception as e:
            logger.error(f"Error generating hashtags: {str(e)}")
            return self._fallback_hashtags(text, random.randint(10, 15))

    def _fallback_hashtags(self, text: str, count: int) -> List[str]:
        generic_tags = [
            "#News", "#IndiaNews", "#LatestNews", "#BreakingNews", "#Trending",
            "#CurrentEvents", "#Headlines", "#TopStories", "#IndiaUpdate", "#DailyNews",
            "#HotNews", "#IndiaToday", "#NewsAlert", "#StoryOfTheDay", "#MustRead"
        ]
        words = [word for word in text.split() if len(word) > 5 and word.isalpha()]
        custom_tags = [f"#{word.capitalize()}" for word in words[:count]]
        all_tags = generic_tags + custom_tags
        return all_tags[:count]

    def _save_thumbnail(self, thumbnail_url: Optional[str]) -> Optional[str]:
        """Download and save a high-quality thumbnail (>50KB), converting WebP to JPG if needed."""
        if not thumbnail_url:
            logger.warning("No thumbnail URL provided")
            return None
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(thumbnail_url, headers=headers, timeout=60)
            if response.status_code != 200:
                logger.error(f"Failed to download thumbnail: HTTP {response.status_code}")
                return None

            content_length = len(response.content)
            if content_length < 51200:  # 50KB minimum
                logger.warning(f"Thumbnail too small ({content_length} bytes), below 50KB threshold")
                return None

            # Check image format
            image_format = imghdr.what(None, response.content)
            if image_format not in ['jpeg', 'png', 'webp']:
                logger.error(f"Invalid image format received: {image_format}")
                return None

            url_hash = hashlib.md5(thumbnail_url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{url_hash}.jpg"
            image_path = os.path.join("images", filename)

            # Convert WebP to JPG if necessary
            if image_format == 'webp':
                img = Image.open(io.BytesIO(response.content)).convert('RGB')
                img.save(image_path, 'JPEG', quality=95)  # High quality save
                logger.info(f"Converted WebP to JPG and saved high-quality thumbnail to {image_path} ({content_length} bytes)")
            else:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Saved high-quality thumbnail to {image_path} ({content_length} bytes)")
            return image_path
        except Exception as e:
            logger.error(f"Error saving thumbnail: {str(e)}")
            return None

    def _is_promotional(self, text: str) -> bool:
        promotional_keywords = ["advertisement", "sponsored", "promotion", "buy now", "limited offer"]
        return any(keyword in text.lower() for keyword in promotional_keywords)

    def _is_astrology(self, title: str, content: str) -> bool:
        """Check if the article is astrology-related."""
        astrology_keywords = ["astrology", "horoscope", "zodiac", "starsign", "planetary", "astrologer"]
        text = (title + " " + content).lower()
        return any(keyword in text for keyword in astrology_keywords)

    def _load_posted_articles(self) -> set:
        if not os.path.exists(self.posted_articles_file):
            return set()
        with open(self.posted_articles_file, 'r') as file:
            return set(line.strip() for line in file)

    def _save_posted_articles(self) -> None:
        with open(self.posted_articles_file, 'w') as file:
            for url in self.posted_articles:
                file.write(f"{url}\n")

    def post_to_facebook(self, article: NewsArticle, processed_content: ProcessedContent) -> bool:
        try:
            if self._is_promotional(processed_content.summary):
                logger.info(f"Skipping promotional article: {article.title}")
                return False

            if not processed_content.hashtags or len(processed_content.hashtags) < 10:
                logger.warning("Insufficient hashtags generated, creating fallback hashtags")
                processed_content.hashtags = self._fallback_hashtags(article.content, random.randint(10, 15))

            if not processed_content.summary:
                logger.warning("Missing summary for the post. Using title-based summary.")
                processed_content.summary = f"Breaking news: {article.title}. More details to follow."

            if not article.source_name or not article.url:
                logger.warning("Missing source information. Skipping post.")
                return False

            if not processed_content.image_path or not os.path.exists(processed_content.image_path):
                logger.error("Image path is invalid or does not exist")
                return False

            hashtag_text = " ".join(processed_content.hashtags)  # Use all hashtags (10-15)
            message = (
                f"{processed_content.summary}\n\n"
                f"{hashtag_text}\n\n"
                f"Source: {article.source_name}\n"
                f"Read more: {article.url}"
            )

            try:
                with open(processed_content.image_path, 'rb') as image_file:
                    image_format = imghdr.what(image_file)
                    if not image_format:
                        logger.error(f"Invalid image format in {processed_content.image_path}")
                        return False
                    image_file.seek(0)
                    self.fb_api.put_photo(
                        image=image_file,
                        message=message,
                        published=True
                    )
                    logger.info(f"Successfully posted recent article with high-quality image: {article.title}")
            except Exception as e:
                logger.error(f"Failed to post to Facebook: {str(e)}")
                return False

            self.posted_articles.add(article.url)
            self._save_posted_articles()
            return True
        except Exception as e:
            logger.error(f"Error posting to Facebook: {str(e)}")
            return False

    def process_news(self) -> None:
        try:
            while True:
                articles = self.get_viral_news()
                if not articles:
                    logger.warning("No recent articles found")
                    break
                for article in articles:
                    if article.url in self.posted_articles:
                        logger.info(f"Article already posted: {article.title}")
                        continue
                    if self._is_astrology(article.title, article.content):
                        logger.info(f"Skipping astrology-related article: {article.title}")
                        continue
                    processed_content = self.process_article(article)
                    if processed_content:
                        if self.post_to_facebook(article, processed_content):
                            return  # Post one article per run
        except Exception as e:
            logger.error(f"Error in news processing: {str(e)}")

def run_automation(config_path: str = "config.yaml"):
    try:
        config = Config(config_path)
        news_bot = NewsAutomation(config_path)
        news_bot.process_news()
    except Exception as e:
        logger.error(f"Error in automation: {str(e)}")
        raise

if __name__ == "__main__":
    run_automation()