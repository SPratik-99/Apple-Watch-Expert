"""
Real Apple Website Scraper
Actually scrapes apple.com for live Apple Watch information
"""
import requests
from bs4 import BeautifulSoup
import logging
import time
import re
from typing import Optional, Dict, List
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class RealAppleWebScraper:
    """Scrapes real Apple Watch information from apple.com"""
    
    def __init__(self):
        self.base_url = "https://www.apple.com"
        self.session = requests.Session()
        
        # Headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
    
    def test_connection(self) -> bool:
        """Test if we can connect to apple.com"""
        try:
            response = self.session.get(f"{self.base_url}/in/apple-watch/", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Apple.com connection test failed: {e}")
            return False
    
    def search_apple_watch_info(self, query: str) -> str:
        """Search for Apple Watch information based on query"""
        try:
            query_lower = query.lower()
            
            # Determine what to scrape based on query
            if any(term in query_lower for term in ["price", "cost", "pricing"]):
                return self._scrape_pricing_info(query_lower)
            
            elif any(term in query_lower for term in ["compare", "vs", "difference"]):
                return self._scrape_comparison_info(query_lower)
            
            elif any(term in query_lower for term in ["features", "specs", "specification"]):
                return self._scrape_features_info(query_lower)
            
            elif any(term in query_lower for term in ["series 9", "s9"]):
                return self._scrape_series9_info()
            
            elif any(term in query_lower for term in ["se", "affordable"]):
                return self._scrape_se_info()
            
            elif any(term in query_lower for term in ["ultra", "adventure"]):
                return self._scrape_ultra_info()
            
            else:
                return self._scrape_general_info()
                
        except Exception as e:
            logger.error(f"Web scraping failed for '{query}': {e}")
            return ""
    
    def _get_cached_or_fetch(self, url: str, cache_key: str) -> Optional[BeautifulSoup]:
        """Get cached content or fetch from web"""
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_time, cached_content = self.cache[cache_key]
            if current_time - cached_time < self.cache_duration:
                return cached_content
        
        # Fetch from web
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                self.cache[cache_key] = (current_time, soup)
                return soup
            else:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def _scrape_pricing_info(self, query: str) -> str:
        """Scrape current Apple Watch pricing"""
        try:
            # Try Indian Apple website first
            soup = self._get_cached_or_fetch(f"{self.base_url}/in/apple-watch/", "apple_watch_in_main")
            
            if not soup:
                return ""
            
            pricing_info = []
            
            # Look for pricing information in various formats
            price_patterns = [
                r'₹\s*(\d+,?\d*)',
                r'From\s*₹\s*(\d+,?\d*)',
                r'Starting\s*at\s*₹\s*(\d+,?\d*)'
            ]
            
            # Extract text and look for prices
            page_text = soup.get_text()
            
            found_prices = []
            for pattern in price_patterns:
                matches = re.findall(pattern, page_text)
                found_prices.extend(matches)
            
            if found_prices:
                # Remove duplicates and format
                unique_prices = list(set(found_prices))
                pricing_info.append(f"Apple Watch pricing found on apple.com/in: {', '.join(unique_prices)}")
            
            # Look for model-specific information
            if "se" in query:
                se_info = self._extract_model_info(soup, ["SE", "Affordable"])
                if se_info:
                    pricing_info.append(f"Apple Watch SE: {se_info}")
            
            if "series" in query or "s9" in query:
                s9_info = self._extract_model_info(soup, ["Series 9", "Series9"])
                if s9_info:
                    pricing_info.append(f"Apple Watch Series 9: {s9_info}")
            
            if "ultra" in query:
                ultra_info = self._extract_model_info(soup, ["Ultra", "Adventure"])
                if ultra_info:
                    pricing_info.append(f"Apple Watch Ultra: {ultra_info}")
            
            return "\n".join(pricing_info) if pricing_info else ""
            
        except Exception as e:
            logger.error(f"Pricing scraping failed: {e}")
            return ""
    
    def _scrape_series9_info(self) -> str:
        """Scrape Series 9 specific information"""
        try:
            soup = self._get_cached_or_fetch(f"{self.base_url}/in/apple-watch-series-9/", "series9_page")
            
            if not soup:
                return ""
            
            info_parts = []
            
            # Extract key features
            features = self._extract_features(soup)
            if features:
                info_parts.append(f"Apple Watch Series 9 features: {', '.join(features[:5])}")
            
            # Extract pricing if available
            prices = self._extract_prices(soup)
            if prices:
                info_parts.append(f"Series 9 pricing: {', '.join(prices)}")
            
            return " | ".join(info_parts) if info_parts else ""
            
        except Exception as e:
            logger.error(f"Series 9 scraping failed: {e}")
            return ""
    
    def _scrape_se_info(self) -> str:
        """Scrape SE specific information"""
        try:
            soup = self._get_cached_or_fetch(f"{self.base_url}/in/apple-watch-se/", "se_page")
            
            if not soup:
                return ""
            
            info_parts = []
            
            # Extract SE-specific information
            features = self._extract_features(soup)
            if features:
                info_parts.append(f"Apple Watch SE features: {', '.join(features[:5])}")
            
            prices = self._extract_prices(soup)
            if prices:
                info_parts.append(f"SE pricing: {', '.join(prices)}")
            
            return " | ".join(info_parts) if info_parts else ""
            
        except Exception as e:
            logger.error(f"SE scraping failed: {e}")
            return ""
    
    def _scrape_ultra_info(self) -> str:
        """Scrape Ultra specific information"""
        try:
            soup = self._get_cached_or_fetch(f"{self.base_url}/in/apple-watch-ultra-2/", "ultra_page")
            
            if not soup:
                return ""
            
            info_parts = []
            
            features = self._extract_features(soup)
            if features:
                info_parts.append(f"Apple Watch Ultra 2 features: {', '.join(features[:5])}")
            
            prices = self._extract_prices(soup)
            if prices:
                info_parts.append(f"Ultra 2 pricing: {', '.join(prices)}")
            
            return " | ".join(info_parts) if info_parts else ""
            
        except Exception as e:
            logger.error(f"Ultra scraping failed: {e}")
            return ""
    
    def _scrape_comparison_info(self, query: str) -> str:
        """Scrape comparison information"""
        try:
            soup = self._get_cached_or_fetch(f"{self.base_url}/in/watch/compare/", "compare_page")
            
            if not soup:
                return ""
            
            # Extract comparison data
            comparison_data = self._extract_comparison_data(soup)
            if comparison_data:
                return f"Apple Watch comparison from apple.com: {comparison_data}"
            
            return ""
            
        except Exception as e:
            logger.error(f"Comparison scraping failed: {e}")
            return ""
    
    def _scrape_features_info(self, query: str) -> str:
        """Scrape general features information"""
        try:
            soup = self._get_cached_or_fetch(f"{self.base_url}/in/apple-watch/", "features_page")
            
            if not soup:
                return ""
            
            features = self._extract_features(soup)
            if features:
                return f"Apple Watch features from apple.com: {', '.join(features[:8])}"
            
            return ""
            
        except Exception as e:
            logger.error(f"Features scraping failed: {e}")
            return ""
    
    def _scrape_general_info(self) -> str:
        """Scrape general Apple Watch information"""
        try:
            soup = self._get_cached_or_fetch(f"{self.base_url}/in/apple-watch/", "general_info")
            
            if not soup:
                return ""
            
            info_parts = []
            
            # Extract general information
            title = soup.find('title')
            if title:
                info_parts.append(f"Current Apple Watch lineup")
            
            features = self._extract_features(soup)
            if features:
                info_parts.append(f"Key features: {', '.join(features[:5])}")
            
            return " | ".join(info_parts) if info_parts else ""
            
        except Exception as e:
            logger.error(f"General info scraping failed: {e}")
            return ""
    
    def _extract_model_info(self, soup: BeautifulSoup, model_keywords: List[str]) -> str:
        """Extract information about specific models"""
        try:
            info_parts = []
            
            # Look for text containing model keywords
            for keyword in model_keywords:
                elements = soup.find_all(text=re.compile(keyword, re.IGNORECASE))
                if elements:
                    # Get parent elements to extract more context
                    for element in elements[:3]:  # Limit to first 3 matches
                        parent = element.parent
                        if parent:
                            text = parent.get_text(strip=True)
                            if len(text) > 10 and len(text) < 200:
                                info_parts.append(text[:100])
            
            return " | ".join(info_parts[:2]) if info_parts else ""
            
        except Exception as e:
            logger.error(f"Model info extraction failed: {e}")
            return ""
    
    def _extract_features(self, soup: BeautifulSoup) -> List[str]:
        """Extract feature information from soup"""
        try:
            features = []
            
            # Common Apple feature keywords to look for
            feature_keywords = [
                "Always-On", "Retina", "GPS", "Cellular", "ECG", "Blood Oxygen",
                "Heart Rate", "Sleep", "Fitness", "Water Resistant", "Battery",
                "Double Tap", "Siri", "Health", "Workout", "Crash Detection"
            ]
            
            page_text = soup.get_text().lower()
            
            for keyword in feature_keywords:
                if keyword.lower() in page_text:
                    features.append(keyword)
            
            return features[:10]  # Return first 10 found features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return []
    
    def _extract_prices(self, soup: BeautifulSoup) -> List[str]:
        """Extract price information from soup"""
        try:
            prices = []
            
            # Look for Indian rupee prices
            price_pattern = r'₹\s*(\d+,?\d*)'
            page_text = soup.get_text()
            
            matches = re.findall(price_pattern, page_text)
            if matches:
                # Clean and deduplicate prices
                cleaned_prices = []
                for price in matches:
                    cleaned_price = price.replace(',', '')
                    if cleaned_price.isdigit():
                        price_int = int(cleaned_price)
                        if 10000 <= price_int <= 200000:  # Reasonable Apple Watch price range
                            cleaned_prices.append(f"₹{price}")
                
                # Remove duplicates while preserving order
                seen = set()
                for price in cleaned_prices:
                    if price not in seen:
                        prices.append(price)
                        seen.add(price)
            
            return prices[:5]  # Return first 5 unique prices
            
        except Exception as e:
            logger.error(f"Price extraction failed: {e}")
            return []
    
    def _extract_comparison_data(self, soup: BeautifulSoup) -> str:
        """Extract comparison information"""
        try:
            # Look for comparison tables or feature lists
            tables = soup.find_all(['table', 'div'], class_=re.compile(r'compare|comparison', re.I))
            
            if tables:
                comparison_text = []
                for table in tables[:2]:  # First 2 comparison elements
                    text = table.get_text(strip=True)
                    if len(text) > 50 and len(text) < 500:
                        comparison_text.append(text[:200])
                
                return " | ".join(comparison_text) if comparison_text else ""
            
            return ""
            
        except Exception as e:
            logger.error(f"Comparison extraction failed: {e}")
            return ""
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        logger.info("Web scraper cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get cache information"""
        current_time = time.time()
        valid_entries = 0
        
        for cache_time, _ in self.cache.values():
            if current_time - cache_time < self.cache_duration:
                valid_entries += 1
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "cache_duration": self.cache_duration
        }