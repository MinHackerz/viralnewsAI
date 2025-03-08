import os
import logging
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, time
from dataclasses import dataclass
from google.generativeai import GenerativeModel, configure
import facebook
import yaml
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import pytz
import hashlib
import imghdr
import random
import time  # Import the time module
from PIL import Image
import io

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

# Configure Google Generative AI
configure(api_key=os.getenv('GOOGLE_API_KEY'))

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

class NewsAPIClient(APIClient):
    """Client for handling news API interactions"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.rate_limit_remaining = 100

    def validate_credentials(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/top-headlines?category=technology&language=en&pageSize=1",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to validate NewsAPI credentials: {str(e)}")
            return False

    def handle_rate_limits(self) -> None:
        if self.rate_limit_remaining <= 0:
            wait_time = self._calculate_wait_time()
            logger.warning(f"Rate limit reached. Waiting for {wait_time} seconds")
            time.sleep(wait_time)

    def _calculate_wait_time(self) -> int:
        return 86400

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
                'summary_length': 60,
                'hashtag_count': 4,
                'image_prompt_prefix': "Professional news illustration of: ",
                'min_image_quality_score': 0.7  # Minimum quality score to accept an image
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
        self.news_api = NewsAPIClient(os.getenv('NEWSAPI_KEY'))
        self.genai_model = GenerativeModel(model_name="gemini-2.0-flash-thinking-exp-1219")
        self.fb_api = facebook.GraphAPI(access_token=os.getenv('FACEBOOK_ACCESS_TOKEN'))
        self.page_id = "530122440176152"

    def _validate_setup(self) -> None:
        # First check environment variables
        required_env_vars = [
            'GOOGLE_API_KEY',
            'FACEBOOK_ACCESS_TOKEN',
            'NEWSAPI_KEY'
        ]

        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing environment variables: {missing_vars}")

        # Then validate credentials
        if not self.news_api.validate_credentials():
            raise ValueError("Invalid NewsAPI credentials")

    def get_viral_news(self) -> List[NewsArticle]:
        self.news_api.handle_rate_limits()
        try:
            response = requests.get(
                f"{self.news_api.base_url}/top-headlines",
                headers={"Authorization": f"Bearer {self.news_api.api_key}"},
                params={
                    "category": "technology",  # Only fetch technology news
                    "language": "en",
                    "pageSize": self.config.config['content']['max_articles']
                },
                timeout=30
            )

            if response.status_code != 200:
                raise Exception(f"NewsAPI error: {response.text}")

            articles = response.json().get('articles', [])
            if not articles:
                logger.warning("No new technology articles found")
                return []

            return [
                NewsArticle(
                    title=article['title'],
                    url=article['url'],
                    content=article.get('description', ''),
                    published_at=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                    source_name=article['source']['name'],
                    thumbnail_url=article.get('urlToImage')
                ) for article in articles
            ]
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []

    def process_article(self, article: NewsArticle) -> Optional[ProcessedContent]:
        try:
            summary = self._generate_summary(article.content)
            hashtags = self._generate_hashtags(article.content)
            image_path = self._save_thumbnail(article.thumbnail_url)

            if not image_path:
                logger.error("Failed to save thumbnail for the article")
                return None

            return ProcessedContent(
                summary=summary,
                hashtags=hashtags,
                image_path=image_path
            )
        except Exception as e:
            logger.error(f"Error processing article: {str(e)}")
            return None

    def _generate_summary(self, text: str) -> str:
        try:
            prompt = f"Summarize this technology news article in exactly {self.config.config['content']['summary_length']} words: {text}"
            response = self.genai_model.generate_content(prompt)
            return response.text if response.text else "Summary not available."
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Summary not available."

    def _generate_hashtags(self, text: str) -> List[str]:
        """Generate hashtags for the news article ensuring proper format and count"""
        try:
            # Technology-focused hashtag prompt
            prompt = (
                f"Create exactly {self.config.config['content']['hashtag_count']} relevant technology hashtags for this news article. "
                f"Each hashtag should start with # and have no spaces. Format each hashtag as #WordWord. "
                f"Focus on tech terms, company names, and industry terms. "
                f"Return ONLY the hashtags and nothing else, separated by spaces: {text}"
            )
            response = self.genai_model.generate_content(prompt)

            if not response.text:
                logger.warning("Empty response from hashtag generation")
                return self._fallback_tech_hashtags(text)

            # Extract hashtags and ensure they start with #
            hashtags = []
            for word in response.text.split():
                word = word.strip()
                if not word.startswith('#'):
                    word = f"#{word}"
                # Remove any punctuation at the end
                if not word[-1].isalnum():
                    word = word[:-1]
                hashtags.append(word)

            # Ensure we have the required number of hashtags
            if len(hashtags) < self.config.config['content']['hashtag_count']:
                # Add tech-specific hashtags if needed
                hashtags.extend(
                    self._fallback_tech_hashtags(text)[:self.config.config['content']['hashtag_count'] - len(hashtags)])

            return hashtags[:self.config.config['content']['hashtag_count']]
        except Exception as e:
            logger.error(f"Error generating hashtags: {str(e)}")
            return self._fallback_tech_hashtags(text)

    def _fallback_tech_hashtags(self, text: str) -> List[str]:
        """Generate fallback technology-focused hashtags if the API call fails"""
        tech_tags = ["#Technology", "#TechNews", "#Innovation", "#Digital", "#Tech", "#AI", "#Cybersecurity",
                     "#Software", "#Hardware", "#DataScience"]

        # Extract potential keywords from the text for additional hashtags
        words = [word for word in text.split() if len(word) > 5 and word.isalpha()]
        custom_tags = [f"#{word.capitalize()}" for word in words[:3]]
        all_tags = tech_tags + custom_tags
        return all_tags[:self.config.config['content']['hashtag_count']]

    def _save_thumbnail(self, thumbnail_url: Optional[str]) -> Optional[str]:
        if not thumbnail_url:
            return None

        try:
            response = requests.get(thumbnail_url, timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to download thumbnail: {response.status_code}")
                return None

            image_format = imghdr.what(None, response.content)
            if image_format not in ['jpeg', 'png']:
                logger.error(f"Invalid image format received: {image_format}")
                return None

            url_hash = hashlib.md5(thumbnail_url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            extension = 'jpg' if image_format == 'jpeg' else image_format
            filename = f"{timestamp}_{url_hash}.{extension}"
            image_path = os.path.join("images", filename)

            with open(image_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Thumbnail saved to {image_path}")
            return image_path
        except Exception as e:
            logger.error(f"Error saving thumbnail: {str(e)}")
            return None

    def _is_promotional(self, text: str) -> bool:
        promotional_keywords = ["advertisement", "sponsored", "promotion", "buy now", "limited offer"]
        return any(keyword in text.lower() for keyword in promotional_keywords)

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

            # Ensure we have hashtags
            if not processed_content.hashtags or len(processed_content.hashtags) == 0:
                logger.warning("No hashtags generated, creating fallback hashtags")
                processed_content.hashtags = self._fallback_tech_hashtags(article.content)

            # Check if all required parts are available
            if not processed_content.summary:
                logger.warning("Missing summary for the post. Generating simple summary.")
                processed_content.summary = f"Latest tech news: {article.title}"

            if not article.source_name or not article.url:
                logger.warning("Missing source information. Skipping post.")
                return False

            # Check for image quality - if no image or quality below threshold, we cannot proceed
            if not processed_content.image_path or not os.path.exists(processed_content.image_path):
                logger.error("Image path is invalid or does not exist")
                return False

            # Format hashtags properly
            hashtag_text = " ".join(processed_content.hashtags[:5])

            # Assemble the complete message with all required components
            message = (
                f"{processed_content.summary}\n\n"
                f"{hashtag_text}\n\n"
                f"Source: {article.source_name}\n"
                f"Read more: {article.url}"
            )

            # Post with high-quality image
            try:
                with open(processed_content.image_path, 'rb') as image_file:
                    # Verify image is readable
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
                    logger.info(f"Successfully posted technology article with high-quality image: {article.title}")
            except Exception as e:
                logger.error(f"Failed to post to Facebook: {str(e)}")
                return False

            # Update posted articles record
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
                    logger.warning("No technology articles found")
                    break

                for article in articles:
                    if article.url in self.posted_articles:
                        logger.info(f"Article already posted: {article.title}")
                        continue

                    processed_content = self.process_article(article)
                    if processed_content:
                        if self.post_to_facebook(article, processed_content):
                            return  # Ensure only one post per run
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