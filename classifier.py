"""
Simple Apple Watch Classifier - No Dependencies
Handles input classification and sentiment analysis
"""
import re
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class SentimentAnalysis:
    """Simple sentiment analysis result"""
    intent: str = "question"
    confidence: float = 0.8
    is_positive: bool = True

class AppleWatchClassifier:
    """Simple, reliable Apple Watch input classifier"""
    
    def __init__(self):
        # Apple Watch related keywords
        self.apple_watch_keywords = [
            "apple watch", "series", "se", "ultra", "watch", "smartwatch",
            "watchos", "apple", "iwatch", "wearable"
        ]
        
        # Budget related keywords
        self.budget_keywords = [
            "budget", "price", "cost", "expensive", "cheap", "afford",
            "rupees", "₹", "money", "worth", "value"
        ]
        
        # Comparison keywords
        self.comparison_keywords = [
            "compare", "vs", "versus", "difference", "better", "best",
            "which", "should i", "recommend", "suggest"
        ]
        
        # Technical support keywords
        self.support_keywords = [
            "problem", "issue", "not working", "broken", "fix", "help",
            "charge", "battery", "connect", "pair", "setup"
        ]
    
    def classify_input(self, text: str) -> Dict[str, any]:
        """Classify user input"""
        text_lower = text.lower()
        
        # Check if it's Apple Watch related
        is_apple_watch = any(keyword in text_lower for keyword in self.apple_watch_keywords)
        
        # Determine category
        category = "general"
        if any(keyword in text_lower for keyword in self.budget_keywords):
            category = "budget"
        elif any(keyword in text_lower for keyword in self.comparison_keywords):
            category = "comparison"
        elif any(keyword in text_lower for keyword in self.support_keywords):
            category = "support"
        
        # Extract budget if present
        budget = self._extract_budget(text)
        
        return {
            "is_apple_watch": is_apple_watch,
            "category": category,
            "budget": budget,
            "confidence": 0.9 if is_apple_watch else 0.3
        }
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Simple sentiment analysis"""
        text_lower = text.lower()
        
        # Determine intent
        if any(word in text_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            intent = "greeting"
        elif any(word in text_lower for word in ["thank", "thanks", "appreciate", "helpful"]):
            intent = "gratitude"
        elif "?" in text or any(word in text_lower for word in ["what", "how", "when", "where", "why", "which"]):
            intent = "question"
        elif any(word in text_lower for word in ["help", "problem", "issue", "not working"]):
            intent = "support"
        else:
            intent = "statement"
        
        # Determine positivity (simple approach)
        positive_words = ["good", "great", "excellent", "love", "like", "best", "amazing", "perfect"]
        negative_words = ["bad", "terrible", "hate", "worst", "problem", "issue", "broken", "not working"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        is_positive = positive_count >= negative_count
        confidence = 0.8
        
        return SentimentAnalysis(
            intent=intent,
            confidence=confidence,
            is_positive=is_positive
        )
    
    def _extract_budget(self, text: str) -> Optional[int]:
        """Extract budget from text"""
        patterns = [
            r'₹\s*(\d+)k',           # ₹30k
            r'₹\s*(\d+),?(\d+)',     # ₹30,000
            r'(\d+)k\s*budget',      # 30k budget
            r'(\d{4,6})\s*rupees',   # 30000 rupees
            r'(\d{4,6})',            # 30000
        ]
        
        text_clean = text.lower().replace(',', '').replace(' ', '')
        
        for pattern in patterns:
            matches = re.findall(pattern, text_clean)
            if matches:
                try:
                    if isinstance(matches[0], tuple):
                        if len(matches[0]) == 2 and matches[0][1]:
                            budget = int(matches[0][0] + matches[0][1])
                        else:
                            budget = int(matches[0][0])
                    else:
                        budget = int(matches[0])
                    
                    # Convert k to thousands
                    if 'k' in text_clean and budget < 1000:
                        budget *= 1000
                    
                    # Reasonable budget range
                    if 10000 <= budget <= 200000:
                        return budget
                        
                except ValueError:
                    continue
        
        return None