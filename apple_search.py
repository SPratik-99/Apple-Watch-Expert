import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import quote_plus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealAppleWebScraper:
    def __init__(self, base_url: str = "https://www.apple.com/in/apple-watch/"):
        self.base_url = base_url

    def test_connection(self) -> bool:
        """Check if Apple.com is reachable."""
        try:
            response = requests.get(self.base_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def search_apple_watch_info(self, query: str) -> str:
        """Scrape Apple.com search results for Apple Watch info."""
        try:
            search_url = f"https://www.apple.com/in/search/{quote_plus(query)}"
            logger.info(f"Scraping Apple.com for query: {query}")
            response = requests.get(search_url, timeout=5)

            if response.status_code != 200:
                return f"Could not fetch info from Apple.com (status: {response.status_code})"

            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("li", {"class": "rf-serp-product"}, limit=3)

            if not results:
                return "No results found on Apple.com."

            scraped_info = []
            for r in results:
                title = r.find("h2").get_text(strip=True) if r.find("h2") else "No title"
                desc = r.find("p").get_text(strip=True) if r.find("p") else "No description"
                scraped_info.append(f"🔹 {title}\n   {desc}")

            return "\n".join(scraped_info)

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return "Error occurred while fetching info from Apple.com."

    def fallback_query(self, query: str) -> str:
        """Fallback mechanism when chatbot has no local data."""
        logger.info(f"No local data found. Falling back to Apple.com for query: '{query}'")

        if not self.test_connection():
            return "Sorry, I’m unable to connect to Apple.com right now."

        raw_response = self.search_apple_watch_info(query)

        if not raw_response.strip():
            return (
                "I couldn’t find specific information on Apple.com either. "
                "Check directly at https://www.apple.com/in/apple-watch/"
            )

        return self._format_response(raw_response, query)

    def _format_response(self, raw: str, query: str) -> str:
        """Make scraped info sound chat-friendly."""
        q = query.lower()

        if any(word in q for word in ["price", "cost", "pricing"]):
            return f"💰 Here’s the latest Apple Watch pricing info from Apple.com:\n{raw}"

        elif any(word in q for word in ["compare", "vs", "difference"]):
            return f"📊 Here’s a quick comparison from Apple.com:\n{raw}"

        elif any(word in q for word in ["feature", "spec"]):
            return f"⚡ Some key Apple Watch features from Apple.com:\n{raw}"

        elif "series 9" in q or "s9" in q:
            return f"⌚ Details about Apple Watch Series 9 from Apple.com:\n{raw}"

        elif "se" in q:
            return f"⌚ Details about Apple Watch SE from Apple.com:\n{raw}"

        elif "ultra" in q or "adventure" in q:
            return f"🏔️ Here’s what Apple.com says about Apple Watch Ultra 2:\n{raw}"

        else:
            return f"📖 Here’s the latest Apple Watch information from Apple.com:\n{raw}"
