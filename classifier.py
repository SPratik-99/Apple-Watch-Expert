"""
Input Classifier and Sentiment Analyzer
Determines if questions are Apple Watch related and analyzes user sentiment
"""
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    emotion: str  # frustrated, excited, curious, neutral, etc.
    intent: str   # question, problem, comparison, buying, etc.
    urgency: str  # low, medium, high
    confidence: float  # 0.0 to 1.0

class AppleWatchClassifier:
    """Classifies questions as Apple Watch related and analyzes sentiment"""
    
    def __init__(self):
        # Apple Watch related keywords
        self.apple_watch_keywords = {
            "direct_mentions": [
                "apple watch", "applewatch", "iwatch", "watch", "smartwatch"
            ],
            "models": [
                "se", "series 9", "series 8", "series 7", "series 6", 
                "ultra", "ultra 2", "edition", "sport", "nike"
            ],
            "features": [
                "heart rate", "ecg", "blood oxygen", "sleep tracking", "fitness",
                "workout", "gps", "cellular", "siri", "apple pay", "activity rings",
                "fall detection", "crash detection", "always on", "digital crown",
                "side button", "band", "strap", "charging", "battery"
            ],
            "health_fitness": [
                "health", "fitness", "exercise", "heart", "pulse", "steps",
                "calories", "workout", "swimming", "running", "cycling"
            ],
            "technical": [
                "setup", "pair", "sync", "connect", "bluetooth", "wifi",
                "update", "reset", "restart", "problem", "issue", "fix"
            ]
        }
        
        # Sentiment patterns
        self.emotion_patterns = {
            "frustrated": [
                r"\b(frustrated|annoyed|angry|upset|hate|terrible|awful|stupid)\b",
                r"\b(doesn't work|not working|broken|useless|crap)\b",
                r"[!]{2,}",  # Multiple exclamation marks
                r"\b(wtf|damn|seriously|ridiculous)\b"
            ],
            "excited": [
                r"\b(love|awesome|amazing|excited|great|fantastic|perfect)\b",
                r"\b(can't wait|so cool|incredible)\b",
                r"[!]+",
                r"😍|🤩|😊|🔥|💯"
            ],
            "curious": [
                r"\b(curious|wondering|interested|want to know)\b",
                r"\b(what if|how about|tell me about)\b"
            ],
            "confused": [
                r"\b(confused|don't understand|not sure|unclear)\b",
                r"\b(help me understand|explain|clarify)\b",
                r"[?]{2,}"
            ]
        }
        
        self.intent_patterns = {
            "buying": [
                r"\b(buy|purchase|get|order|price|cost|budget|worth|best|recommend)\b"
            ],
            "comparing": [
                r"\b(vs|versus|compare|comparison|difference|which|better)\b"
            ],
            "troubleshooting": [
                r"\b(problem|issue|fix|broken|not working|error|trouble)\b"
            ],
            "learning": [
                r"\b(what|how|why|explain|tell me|learn|understand)\b"
            ],
            "setup": [
                r"\b(setup|set up|pair|connect|sync|install|configure)\b"
            ]
        }
    
    def classify_input(self, user_input: str) -> Dict:
        """Classify if input is Apple Watch related"""
        if not user_input or len(user_input.strip()) < 2:
            return {"is_apple_watch": False, "confidence": 0.0, "reason": "Empty input"}
        
        input_lower = user_input.lower().strip()
        
        # Check for Apple Watch keywords
        score = 0
        
        # Direct mentions (high weight)
        for keyword in self.apple_watch_keywords["direct_mentions"]:
            if keyword in input_lower:
                score += 1.0
                break
        
        # Model mentions
        for model in self.apple_watch_keywords["models"]:
            if model in input_lower:
                score += 0.7
                break
        
        # Feature mentions
        for feature in self.apple_watch_keywords["features"]:
            if feature in input_lower:
                score += 0.5
                break
        
        # Health/fitness mentions
        for term in self.apple_watch_keywords["health_fitness"]:
            if term in input_lower:
                score += 0.3
                break
        
        # Technical mentions
        for term in self.apple_watch_keywords["technical"]:
            if term in input_lower:
                score += 0.2
                break
        
        # Determine if Apple Watch related
        is_apple_watch = score >= 0.5
        confidence = min(score, 1.0)
        
        return {
            "is_apple_watch": is_apple_watch,
            "confidence": confidence,
            "reason": f"Keyword score: {score:.1f}"
        }
    
    def analyze_sentiment(self, user_input: str) -> SentimentAnalysis:
        """Analyze sentiment of user input"""
        if not user_input:
            return SentimentAnalysis("neutral", "general", "low", 0.0)
        
        input_lower = user_input.lower().strip()
        
        # Detect emotion
        emotion = "neutral"
        for emotion_type, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    emotion = emotion_type
                    break
            if emotion != "neutral":
                break
        
        # Detect intent
        intent = "general"
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    intent = intent_type
                    break
            if intent != "general":
                break
        
        # Special cases for intent
        if any(word in input_lower for word in ["hi", "hello", "hey"]):
            intent = "greeting"
        elif any(word in input_lower for word in ["thank", "thanks"]):
            intent = "gratitude"
        
        # Detect urgency
        urgency = "low"
        if any(word in input_lower for word in ["urgent", "emergency", "asap", "now", "immediately"]):
            urgency = "high"
        elif any(word in input_lower for word in ["problem", "broken", "not working", "help"]):
            urgency = "medium"
        
        # Calculate confidence
        confidence = 0.7  # Base confidence
        if emotion != "neutral":
            confidence += 0.1
        if intent != "general":
            confidence += 0.1
        if "?" in user_input:
            confidence += 0.1
        
        return SentimentAnalysis(emotion, intent, urgency, min(confidence, 1.0))
    
    def is_casual_greeting(self, user_input: str) -> bool:
        """Check if input is a casual greeting"""
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        return any(greeting in user_input.lower() for greeting in greetings)
    
    def is_gratitude(self, user_input: str) -> bool:
        """Check if input expresses gratitude"""
        gratitude_words = ["thank", "thanks", "thx", "appreciate", "grateful"]
        return any(word in user_input.lower() for word in gratitude_words)

# Global classifier instance
classifier = AppleWatchClassifier()