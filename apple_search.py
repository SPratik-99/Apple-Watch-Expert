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

logger = logging.getLogger(__name__)

class RealAppleWebScraper:
    """Scrapes real Apple Watch information from apple.com"""
    
    def __init__(self):
        self.session = requests.Session()
        
        # Headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.apple_urls = {
            "main_page": "https://www.apple.com/in/apple-watch/",
            "compare": "https://www.apple.com/in/apple-watch/compare/",
            "se": "https://www.apple.com/in/apple-watch-se/",
            "series_9": "https://www.apple.com/in/apple-watch-series-9/", 
            "ultra": "https://www.apple.com/in/apple-watch-ultra-2/",
            "specs": "https://www.apple.com/in/apple-watch/specs/",
            "health": "https://www.apple.com/in/apple-watch/health/",
        } 
        
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
    
    def test_connection(self) -> bool:
        """Test if we can connect to apple.com"""
        try:
            response = self.session.get(self.apple_urls["main_page"], timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Apple.com connection test failed: {e}")
            return False
    
    def search_apple_watch_info(self, query: str) -> str:
        """Search for Apple Watch information based on query"""
        try:
            query_lower = query.lower()
            
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
        
        if cache_key in self.cache:
            cached_time, cached_content = self.cache[cache_key]
            if current_time - cached_time < self.cache_duration:
                return cached_content
        
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
            soup = self._get_cached_or_fetch(self.apple_urls["main_page"], "apple_watch_main")
            if not soup:
                return ""
            
            pricing_info = []
            price_patterns = [r'₹\s*(\d+,?\d*)']
            page_text = soup.get_text()
            
            found_prices = []
            for pattern in price_patterns:
                matches = re.findall(pattern, page_text)
                found_prices.extend(matches)
            
            if found_prices:
                unique_prices = list(set(found_prices))
                pricing_info.append(f"Apple Watch pricing on apple.com: {', '.join(unique_prices)}")
            
            return "\n".join(pricing_info) if pricing_info else ""
            
        except Exception as e:
            logger.error(f"Pricing scraping failed: {e}")
            return ""
    
    def _scrape_series9_info(self) -> str:
        """Scrape Series 9 specific information"""
        try:
            soup = self._get_cached_or_fetch(self.apple_urls["series_9"], "series9_page")
            if not soup:
                return ""
            
            features = self._extract_features(soup)
            prices = self._extract_prices(soup)
            
            info = []
            if features:
                info.append(f"Apple Watch Series 9 features: {', '.join(features[:5])}")
            if prices:
                info.append(f"Series 9 pricing: {', '.join(prices)}")
            return " | ".join(info)
        except Exception as e:
            logger.error(f"Series 9 scraping failed: {e}")
            return ""
    
    def _scrape_se_info(self) -> str:
        """Scrape SE specific information"""
        try:
            soup = self._get_cached_or_fetch(self.apple_urls["se"], "se_page")
            if not soup:
                return ""
            features = self._extract_features(soup)
            prices = self._extract_prices(soup)
            info = []
            if features:
                info.append(f"Apple Watch SE features: {', '.join(features[:5])}")
            if prices:
                info.append(f"SE pricing: {', '.join(prices)}")
            return " | ".join(info)
        except Exception as e:
            logger.error(f"SE scraping failed: {e}")
            return ""
    
    def _scrape_ultra_info(self) -> str:
        """Scrape Ultra specific information"""
        try:
            soup = self._get_cached_or_fetch(self.apple_urls["ultra"], "ultra_page")
            if not soup:
                return ""
            features = self._extract_features(soup)
            prices = self._extract_prices(soup)
            info = []
            if features:
                info.append(f"Apple Watch Ultra 2 features: {', '.join(features[:5])}")
            if prices:
                info.append(f"Ultra 2 pricing: {', '.join(prices)}")
            return " | ".join(info)
        except Exception as e:
            logger.error(f"Ultra scraping failed: {e}")
            return ""
    
    def _scrape_comparison_info(self, query: str) -> str:
        """Scrape comparison information"""
        try:
            soup = self._get_cached_or_fetch(self.apple_urls["compare"], "compare_page")
            if not soup:
                return ""
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
            soup = self._get_cached_or_fetch(self.apple_urls["main_page"], "features_page")
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
            soup = self._get_cached_or_fetch(self.apple_urls["main_page"], "general_info")
            if not soup:
                return ""
            features = self._extract_features(soup)
            if features:
                return f"Current Apple Watch lineup | Key features: {', '.join(features[:5])}"
            return ""
        except Exception as e:
            logger.error(f"General info scraping failed: {e}")
            return ""
    
    # --- helper extractors ---
    def _extract_features(self, soup: BeautifulSoup) -> List[str]:
        try:
            features = []
            keywords = [
                "Always-On", "Retina", "GPS", "Cellular", "ECG", "Blood Oxygen",
                "Heart Rate", "Sleep", "Fitness", "Water Resistant", "Battery",
                "Double Tap", "Siri", "Health", "Workout", "Crash Detection"
            ]
            page_text = soup.get_text().lower()
            for kw in keywords:
                if kw.lower() in page_text:
                    features.append(kw)
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return []
    
    def _extract_prices(self, soup: BeautifulSoup) -> List[str]:
        try:
            prices = []
            matches = re.findall(r'₹\s*(\d+,?\d*)', soup.get_text())
            for price in matches:
                price_int = int(price.replace(',', ''))
                if 10000 <= price_int <= 200000:
                    prices.append(f"₹{price}")
            return list(dict.fromkeys(prices))  # dedupe
        except Exception as e:
            logger.error(f"Price extraction failed: {e}")
            return []
    
    def _extract_comparison_data(self, soup: BeautifulSoup) -> str:
        try:
            sections = soup.find_all(['table','div'], class_=re.compile(r'compare|comparison', re.I))
            if not sections:
                return ""
            texts = []
            for sec in sections[:2]:
                txt = sec.get_text(strip=True)
                if 50 < len(txt) < 500:
                    texts.append(txt[:200])
            return " | ".join(texts)
        except Exception as e:
            logger.error(f"Comparison extraction failed: {e}")
            return ""
    
    def clear_cache(self):
        self.cache.clear()
        logger.info("Web scraper cache cleared")
    
    def get_cache_info(self) -> Dict:
        current_time = time.time()
        valid_entries = sum(1 for t,_ in self.cache.values() if current_time - t < self.cache_duration)
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "cache_duration": self.cache_duration
        }
