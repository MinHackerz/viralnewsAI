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
import time
from PIL import Image
import io
import random
import google.generativeai as genai

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

def get_image_format(file_or_bytes):
    """Replacement for imghdr functionality using PIL"""
    try:
        image = Image.open(file_or_bytes)
        return image.format.lower() if image.format else None
    except Exception:
        return None

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
        required_env_vars = ['FACEBOOK_ACCESS_TOKEN', 'NEWSAPI_KEY', 'GOOGLE_API_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing environment variables: {missing_vars}")
        if not self.news_api.validate_credentials():
            raise ValueError("Invalid NewsData.io credentials")
        
        # Configure Gemini API
        try:
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            logger.info("Successfully configured Google Generative AI")
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {str(e)}")
            raise ValueError(f"Failed to configure Gemini API: {str(e)}")

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
            # Make sure content is not None before processing
            article_content = article.content if article.content is not None else ""
            
            # Generate summary with safe content
            summary = self._generate_summary(article.title, article_content)
            
            # Generate hashtags with safe content
            hashtags = self._generate_hashtags(article_content)
            
            # Get image
            image_path = self._save_thumbnail(article.thumbnail_url)
            if not image_path:
                logger.error("Failed to save high-quality thumbnail for the article")
                return None
                
            return ProcessedContent(summary=summary, hashtags=hashtags, image_path=image_path)
        except Exception as e:
            logger.error(f"Error processing article: {str(e)}")
            return None

    def _generate_summary(self, title: str, content: str) -> str:
        """Generate a comprehensive summary of the article using Gemini API."""
        try:
            # Ensure title and content are not None
            safe_title = title if title else "Untitled Article"
            safe_content = content if content else "No content available"
            
            # Configure the model - use the correct model name
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # Prepare prompt
                prompt = f"""
                Please create a comprehensive news summary based on this information:
                
                Title: {safe_title}
                
                Content: {safe_content}
                
                Generate a detailed summary that captures the key points of this news article.
                Write in a journalistic, informative style. Do not add any fictional elements.
                Focus on facts presented in the article. The summary should be comprehensive 
                with no word limit constraints.
                """
                
                # Generate response from Gemini
                response = model.generate_content(prompt)
                
                # Get the summary from the response
                if response and hasattr(response, 'text'):
                    summary = response.text.strip()
                    if summary:
                        logger.info(f"Generated summary using Gemini API: {summary[:100]}...")
                        return summary
            except Exception as e:
                logger.error(f"Error with Gemini API call: {str(e)}")
                
            # If we get here, something went wrong with the API call
            return f"Breaking news: {safe_title}."
            
        except Exception as e:
            logger.error(f"Error generating summary with Gemini API: {str(e)}")
            # We'll still return something rather than None to prevent downstream issues
            return f"Breaking news: {title if title else 'Recent News Update'}."

    def _generate_hashtags(self, text: str) -> List[str]:
        try:
            # Ensure text is not None
            safe_text = text if text else ""
            
            hashtag_count = random.randint(10, 15)  # Rotate between 10-15
            # Simple hashtag generation from text
            words = [w.capitalize() for w in safe_text.split() if len(w) > 4 and w.isalpha()]
            hashtags = [f"#{w}" for w in words[:hashtag_count]]
            if len(hashtags) < hashtag_count:
                hashtags.extend(self._fallback_hashtags(safe_text, hashtag_count - len(hashtags)))
            return hashtags[:hashtag_count]
        except Exception as e:
            logger.error(f"Error generating hashtags: {str(e)}")
            return self._fallback_hashtags(text if text else "", random.randint(10, 15))

    def _fallback_hashtags(self, text: str, count: int) -> List[str]:
        generic_tags = [
            "#News", "#IndiaNews", "#LatestNews", "#BreakingNews", "#Trending",
            "#CurrentEvents", "#Headlines", "#TopStories", "#IndiaUpdate", "#DailyNews",
            "#HotNews", "#IndiaToday", "#NewsAlert", "#StoryOfTheDay", "#MustRead"
        ]
        
        # Ensure text is not None
        safe_text = text if text is not None else ""
        
        words = [word for word in safe_text.split() if len(word) > 5 and word.isalpha()]
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
            content_bytes = io.BytesIO(response.content)
            image_format = get_image_format(content_bytes)
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
        # Ensure text is not None
        if text is None:
            return False
            
        promotional_keywords = ["advertisement", "sponsored", "promotion", "buy now", "limited offer"]
        return any(keyword in text.lower() for keyword in promotional_keywords)

    def _is_astrology(self, title: str, content: str) -> bool:
        """Check if the article is astrology-related."""
        try:
            # Ensure title and content are not None
            safe_title = title if title is not None else ""
            safe_content = content if content is not None else ""
            
            combined_text = (safe_title + " " + safe_content).lower()
            astrology_keywords = ["astrology", "horoscope", "zodiac", "starsign", "planetary", "astrologer"]
            return any(keyword in combined_text for keyword in astrology_keywords)
        except Exception as e:
            logger.error(f"Error checking for astrology content: {str(e)}")
            return False

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
                logger.warning("Missing summary from Gemini API, using article title.")
                processed_content.summary = f"Breaking news: {article.title}."

            if not article.source_name or not article.url:
                logger.warning("Missing source information. Skipping post.")
                return False

            if not processed_content.image_path or not os.path.exists(processed_content.image_path):
                logger.error("Image path is invalid or does not exist")
                return False

            hashtag_text = " ".join(processed_content.hashtags)
            message = (
                f"{processed_content.summary}\n\n"
                f"{hashtag_text}\n\n"
                f"Source: {article.source_name}\n"
                f"Read more: {article.url}"
            )

            try:
                with open(processed_content.image_path, 'rb') as image_file:
                    image_format = get_image_format(image_file)
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
                logger.info("Starting to fetch and process news articles")
                articles = self.get_viral_news()
                
                if not articles:
                    logger.warning("No recent articles found")
                    break
                
                logger.info(f"Processing {len(articles)} articles")
                for i, article in enumerate(articles):
                    try:
                        logger.info(f"Processing article {i+1}/{len(articles)}: {article.title}")
                        
                        # Check if already posted
                        if article.url in self.posted_articles:
                            logger.info(f"Article already posted: {article.title}")
                            continue
                            
                        # Check for astrology (safely)
                        if article.title is None and article.content is None:
                            logger.warning("Article has no title or content, skipping")
                            continue
                            
                        # Check if article is about astrology
                        try:
                            if self._is_astrology(article.title, article.content):
                                logger.info(f"Skipping astrology-related article: {article.title}")
                                continue
                        except Exception as e:
                            logger.error(f"Error during astrology check: {str(e)}")
                            continue
                        
                        # Process the article
                        logger.info(f"Processing content for article: {article.title}")
                        processed_content = self.process_article(article)
                        
                        if processed_content:
                            logger.info(f"Successfully processed article: {article.title}")
                            logger.info(f"Attempting to post to Facebook: {article.title}")
                            
                            if self.post_to_facebook(article, processed_content):
                                logger.info(f"Successfully posted article: {article.title}")
                                return  # Post one article per run
                            else:
                                logger.warning(f"Failed to post article: {article.title}")
                        else:
                            logger.warning(f"Failed to process article: {article.title}")
                    except Exception as e:
                        logger.error(f"Error processing article {article.title}: {str(e)}")
                        continue
                
                # If we reach here, we couldn't post any articles
                logger.info("No articles were posted in this run")
                break
                
        except Exception as e:
            logger.error(f"Error in news processing: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

def test_gemini():
    """Test function to verify Gemini API connectivity."""
    try:
        # Configure the API
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("Error: GOOGLE_API_KEY not found in environment variables")
            return False
            
        genai.configure(api_key=api_key)
        
        # Try a simple prompt
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Write a one-sentence test response.")
        
        if response and hasattr(response, 'text'):
            print(f"Gemini API Test Success: {response.text}")
            return True
        else:
            print("Error: Unexpected response format from Gemini API")
            return False
    except Exception as e:
        print(f"Error testing Gemini API: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def run_automation(config_path: str = "config.yaml"):
    try:
        logger.info("Starting News Automation")
        config = Config(config_path)
        news_bot = NewsAutomation(config_path)
        news_bot.process_news()
        logger.info("News Automation completed successfully")
    except Exception as e:
        logger.error(f"Error in automation: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "test_gemini":
        print("Testing Gemini API connectivity...")
        if test_gemini():
            print("Gemini API test passed!")
            sys.exit(0)
        else:
            print("Gemini API test failed!")
            sys.exit(1)
    
    # Normal execution
    try:
        logger.info("Starting ViralNewsAI application")
        run_automation()
        logger.info("ViralNewsAI application completed successfully")
    except Exception as e:
        logger.error(f"Critical error in ViralNewsAI application: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)