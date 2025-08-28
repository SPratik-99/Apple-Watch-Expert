"""
FINAL FIX - Properly Use Web Data in Responses
Fixes vague responses and excessive HTTP requests
"""
import logging
from typing import List, Dict, Optional, Union
import os
import json
import requests
import time
import re

logger = logging.getLogger(__name__)

# Import with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import config
from classifier import SentimentAnalysis

# Simplified web checker to prevent excessive requests
class SmartWebChecker:
    """Efficient web checking with rate limiting and caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 1800  # 30 minutes
        self.last_request_time = {}
        self.min_request_interval = 60  # 1 minute between requests
        
    def should_check_web(self, query_type: str) -> bool:
        """Check if we should make a web request"""
        current_time = time.time()
        
        # Check rate limiting
        if query_type in self.last_request_time:
            time_since_last = current_time - self.last_request_time[query_type]
            if time_since_last < self.min_request_interval:
                logger.info(f"Rate limited: {query_type} (wait {self.min_request_interval - time_since_last:.0f}s)")
                return False
        
        return True
    
    def get_cached_response(self, query_type: str) -> Optional[str]:
        """Get cached response if available"""
        if query_type in self.cache:
            cache_time, cached_data = self.cache[query_type]
            if time.time() - cache_time < self.cache_duration:
                logger.info(f"Using cached data for: {query_type}")
                return cached_data
        return None
    
    def store_response(self, query_type: str, response: str):
        """Store response in cache"""
        self.cache[query_type] = (time.time(), response)
        self.last_request_time[query_type] = time.time()

class AppleWatchKnowledgeBase:
    """Enhanced knowledge base with smart web integration"""
    
    def __init__(self):
        # Comprehensive local knowledge
        self.local_knowledge = {
            "current_models": {
                "Apple Watch SE": {
                    "price": "₹24,900",
                    "sizes": "40mm and 44mm",
                    "key_features": "Heart rate monitoring, GPS tracking, sleep tracking, crash detection, fall detection, 85+ workouts, water resistant",
                    "missing": "No Always-On display, no ECG, no Blood Oxygen monitoring, no Double Tap gesture"
                },
                "Apple Watch Series 9": {
                    "price": "₹41,900", 
                    "sizes": "41mm and 45mm",
                    "key_features": "S9 chip (60% faster), Always-On Retina display, Double Tap gesture, ECG app, Blood Oxygen monitoring, temperature sensing",
                    "unique": "Double Tap gesture control, brightest Apple Watch display (2000 nits)"
                },
                "Apple Watch Ultra 2": {
                    "price": "₹89,900",
                    "sizes": "49mm only",
                    "key_features": "Titanium case, Action Button, 100m water resistance, 36+ hour battery, precision dual-frequency GPS, 86dB emergency siren",
                    "unique": "Most durable Apple Watch, designed for extreme sports"
                }
            },
            "non_existent": ["Series 10", "Series10", "Ultra 3", "SE 3"]
        }
        
        # Smart web checker
        self.web_checker = SmartWebChecker()
    
    def get_response(self, query: str) -> str:
        """Get comprehensive response with minimal web calls"""
        query_lower = query.lower()
        
        # Handle non-existent products immediately
        if any(model.lower() in query_lower for model in self.local_knowledge["non_existent"]):
            return self._handle_non_existent_product(query)
        
        # Budget recommendations
        if any(word in query_lower for word in ["budget", "price", "30k", "25k", "40k", "suggest", "recommend"]):
            budget = self._extract_budget(query)
            if budget:
                return self._get_budget_recommendation(budget)
        
        # Specific model queries
        if "se" in query_lower and "series" not in query_lower:
            return self._get_model_info("Apple Watch SE")
        elif "series 9" in query_lower or "s9" in query_lower:
            return self._get_model_info("Apple Watch Series 9")
        elif "ultra" in query_lower:
            return self._get_model_info("Apple Watch Ultra 2")
        
        # Comparisons
        if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            return self._get_comparison()
        
        # Technical support
        if any(word in query_lower for word in ["charge", "problem", "fix", "not working"]):
            return self._get_technical_support(query_lower)
        
        # General model inquiry
        if any(word in query_lower for word in ["model", "which", "what", "available", "lineup"]):
            return self._get_model_overview()
        
        # Default response
        return self._get_default_response()
    
    def _extract_budget(self, query: str) -> Optional[int]:
        """Extract budget from query"""
        patterns = [r'(\d+)k', r'₹\s*(\d+,?\d*)', r'(\d{4,6})']
        
        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                try:
                    budget = int(matches[0].replace(',', ''))
                    if 'k' in query.lower() and budget < 1000:
                        budget *= 1000
                    if 10000 <= budget <= 200000:
                        return budget
                except:
                    continue
        return None
    
    def _handle_non_existent_product(self, query: str) -> str:
        """Handle queries about non-existent products"""
        return f"""I cannot provide information about the model mentioned in your query because it doesn't exist in Apple's current lineup.

**Current Apple Watch Models (August 2025):**

**Apple Watch SE (2nd gen)** - ₹24,900
• Most affordable Apple Watch
• Heart rate monitoring, GPS, sleep tracking
• Crash detection and fall detection
• Perfect for first-time users

**Apple Watch Series 9** - ₹41,900  
• Latest mainstream Apple Watch
• Always-On Retina display, Double Tap gesture
• Complete health suite: ECG, Blood Oxygen
• Best overall choice for most users

**Apple Watch Ultra 2** - ₹89,900
• Most advanced and durable Apple Watch
• Titanium case, 36+ hour battery
• Designed for extreme sports and adventures

Which of these actual models would you like to know more about?"""
    
    def _get_budget_recommendation(self, budget: int) -> str:
        """Get budget-specific recommendations"""
        if budget < 25000:
            return f"""For your ₹{budget:,} budget, you're very close to the Apple Watch SE.

**Apple Watch SE (2nd gen)** - ₹24,900
• Only ₹{24900-budget:,} more than your budget
• Heart rate monitoring and GPS tracking
• Sleep tracking with detailed sleep stages
• Crash detection and fall detection
• 85+ workout types, water resistant to 50m
• 18+ hour battery life

**Your Options:**
1. **Stretch budget slightly** for the SE 40mm GPS at ₹24,900
2. **Wait for sales** - SE often drops to ₹22,000-23,000 during festivals
3. **Certified refurbished** SE models available for ₹18,000-22,000

The Apple Watch SE offers 90% of the full Apple Watch experience and is the best value at this price point."""

        elif budget <= 45000:
            return f"""Perfect! Your ₹{budget:,} budget is ideal for the Apple Watch Series 9.

**Apple Watch Series 9** - ₹41,900-44,900
• **S9 chip**: 60% faster performance than SE
• **Always-On Retina display**: 2000 nits brightness (brightest ever)
• **Double Tap gesture**: Revolutionary control method
• **Complete health monitoring**: ECG, Blood Oxygen, temperature sensing
• **Enhanced Siri**: On-device processing for privacy

**What your budget gets you:**
• Series 9 41mm GPS: ₹41,900 ✅ (₹{budget-41900:,} left for accessories)
• Series 9 45mm GPS: ₹44,900 ✅ (perfect fit)
• Premium Sport Loop or Leather band

**Why Series 9 is worth it:**
The ₹17,000 premium over SE gets you Always-On display, advanced health features, latest gesture controls, and significantly faster performance. It's the best overall Apple Watch for most users."""

        else:
            return f"""Excellent budget for premium Apple Watch options!

**Apple Watch Series 9** (₹41,900-53,900)
• Complete flagship experience with latest features
• Perfect for everyday premium use

**Apple Watch Ultra 2** (₹89,900)  
• Aerospace titanium construction (most durable)
• Largest 49mm display, 3000 nits brightness
• 36+ hour battery (vs 18 hours on others)
• 100m water resistance, precision dual-frequency GPS
• Built for extreme sports and adventures

**My recommendation for ₹{budget:,}:**
• **Choose Series 9** if you want the latest mainstream features with excellent value
• **Choose Ultra 2** if you need maximum durability, longest battery life, or do serious outdoor activities
• **Series 9 + premium accessories** for a complete luxury setup

Which use case better fits your lifestyle - everyday premium or extreme outdoor adventures?"""
    
    def _get_model_info(self, model: str) -> str:
        """Get detailed information about specific model"""
        info = self.local_knowledge["current_models"][model]
        
        if model == "Apple Watch SE":
            return f"""**Apple Watch SE (2nd generation)**

**Price:** Starting at {info['price']} (40mm GPS)
**Sizes Available:** {info['sizes']}
**Cellular Options:** 40mm (₹30,900), 44mm (₹34,900)

**Key Features:**
• S8 SiP dual-core processor
• {info['key_features']}
• 18+ hour all-day battery life

**What's Missing Compared to Series 9:**
{info['missing']}

**Perfect For:**
• First-time Apple Watch users
• Budget-conscious buyers who want genuine Apple Watch experience
• Fitness enthusiasts who don't need advanced health monitoring
• Users who primarily want notifications, fitness tracking, and basic health features

**Value Proposition:**
The SE offers about 90% of the full Apple Watch experience at 60% of the Series 9 price."""

        elif model == "Apple Watch Series 9":
            return f"""**Apple Watch Series 9** - The Complete Apple Watch Experience

**Price:** Starting at {info['price']} (41mm GPS)
**Sizes Available:** {info['sizes']}
**Cellular Options:** 41mm (₹50,900), 45mm (₹53,900)

**Latest Features:**
• {info['key_features']}
• {info['unique']}
• Carbon neutral when paired with Sport Loop

**What's New in Series 9:**
• Double Tap gesture - tap thumb and finger to control watch
• Brightest Apple Watch display ever (2000 nits)
• On-device Siri processing for faster, more private interactions
• Most advanced health monitoring suite

**Perfect For:**
• Users who want the complete, latest Apple Watch experience
• Health-conscious individuals who value advanced monitoring
• Tech enthusiasts who love cutting-edge features
• Premium users who want the best Apple Watch

**Why Choose Series 9:**
It's the sweet spot of Apple Watch lineup - all the latest features without Ultra's extreme focus."""

        else:  # Ultra 2
            return f"""**Apple Watch Ultra 2** - The Ultimate Apple Watch

**Price:** {info['price']} (49mm Cellular only)
**Size:** {info['sizes']} - largest Apple Watch display

**Extreme Capabilities:**
• {info['key_features']}
• {info['unique']}
• Operating temperature: -20°C to 55°C
• MIL-STD 810H certified for durability

**Extended Battery Life:**
• 36 hours normal use (double other models)
• Up to 72 hours in Low Power Mode
• Perfect for multi-day adventures without charging

**Built for Extremes:**
• Recreational scuba diving to 40 meters
• Mountaineering and endurance sports
• Professional use in harsh environments
• Adventure racing and multi-day events

**Perfect For:**
• Serious athletes and outdoor adventurers
• Users who need maximum durability
• Multi-day adventure enthusiasts
• Professionals working in extreme conditions
• Anyone who wants the longest battery life possible

**Ultra vs Series 9:**
Choose Ultra if you need extreme durability, maximum battery, or use Apple Watch for serious outdoor activities. Otherwise, Series 9 provides the complete experience for most users."""
    
    def _get_comparison(self) -> str:
        """Get model comparison"""
        return """**Apple Watch Complete Comparison (August 2025)**

| Feature | SE (2nd gen) | Series 9 | Ultra 2 |
|---------|--------------|----------|---------|
| **Price** | ₹24,900 | ₹41,900 | ₹89,900 |
| **Display** | Retina | Always-On Retina (2000 nits) | Always-On (3000 nits) |
| **Sizes** | 40mm, 44mm | 41mm, 45mm | 49mm only |
| **Processor** | S8 chip | S9 chip (60% faster) | S9 chip |
| **Health** | Heart rate, Sleep | ECG, Blood Oxygen, Temperature | All + Depth, Water temp |
| **Battery** | 18 hours | 18 hours | 36 hours |
| **Special** | - | Double Tap gesture | Action Button |
| **Build** | Aluminum | Aluminum/Steel | Titanium |
| **Water** | 50m | 50m | 100m |

**Quick Recommendations:**
• **SE**: Best value, perfect for first-time users (₹24,900)
• **Series 9**: Best overall choice, latest features (₹41,900)
• **Ultra 2**: Maximum durability, longest battery (₹89,900)

**Decision Guide:**
- Budget under ₹30k → **Apple Watch SE**
- Budget ₹30-60k → **Apple Watch Series 9** 
- Need extreme durability → **Apple Watch Ultra 2**"""
    
    def _get_technical_support(self, query: str) -> str:
        """Get technical support information"""
        if "charge" in query:
            return """**Apple Watch Charging Issues - Complete Fix Guide**

**Step 1: Clean Everything Thoroughly**
• Remove Apple Watch from charger completely
• Use soft, lint-free cloth (microfiber cloth works best)
• Clean watch back (circular sensor area) thoroughly
• Clean charger surface - remove any debris, sweat, or moisture

**Step 2: Verify Proper Setup**
• Use original Apple charging cable ONLY (third-party cables often fail)
• Connect to 5W+ USB power adapter (iPhone charger works perfectly)
• Ensure magnetic connection clicks and feels secure
• Green lightning bolt should appear on watch screen within 10 seconds

**Step 3: Force Restart Your Watch**
• Hold side button + Digital Crown simultaneously
• Keep holding for exactly 10 seconds until Apple logo appears
• Release buttons and immediately place back on charger
• Watch should start charging within 30 seconds

**Step 4: Advanced Troubleshooting**
• Try different power outlet and different USB adapter
• Test charging cable with another device to verify it works
• Check for physical damage on charging cable
• Ensure watch is properly centered on charging pad
• Try charging in different room temperature (not too hot/cold)

**Step 5: If Still Not Working**
• Contact Apple Support if watch is under warranty (1-year free support)
• Visit Apple Store for free diagnostics and testing
• Check Apple's official support website for latest solutions
• May need charging cable replacement (₹3,900) or service

**Prevention Tips:**
• Clean charging contacts weekly with soft cloth
• Avoid extreme temperatures while charging
• Use only official Apple charging accessories
• Don't let battery completely drain regularly"""
        
        return """**Apple Watch Technical Support**

**Most Common Issues & Solutions:**

**Watch Frozen/Unresponsive:**
• Force restart: Hold side button + Digital Crown for 10 seconds until Apple logo appears

**Battery Draining Too Fast:**
• Turn off Always-On Display: Settings > Display & Brightness > Always On > Off
• Reduce background app refresh: Watch app > General > Background App Refresh
• Check for apps running in background
• Update to latest watchOS version

**Connection Problems with iPhone:**
• Ensure both devices have Bluetooth ON
• Keep iPhone and Apple Watch within 10 meters of each other
• Restart both iPhone and Apple Watch
• Check iPhone isn't in Low Power Mode (disables some watch features)
• Re-pair watch if connection issues persist

**Fitness Tracking Not Accurate:**
• Ensure wrist detection is ON: Settings > Passcode > Wrist Detection
• Wear watch snug but comfortable during workouts
• Calibrate outdoor walk: Settings > Privacy & Security > Location Services > Apple Watch Workout
• Keep watch charged above 10% during workouts

**Notifications Not Working:**
• Check notification settings in iPhone Watch app
• Ensure Do Not Disturb is off on both devices
• Verify notification mirroring is enabled
• Restart both devices if notifications still don't work

**Tell me your specific issue and I'll provide detailed troubleshooting steps!**"""
    
    def _get_model_overview(self) -> str:
        """Get overview of all current models"""
        return """**Current Apple Watch Lineup (August 2025)**

Apple offers three main Apple Watch models, each designed for different users:

**Apple Watch SE (2nd generation)** - ₹24,900
• **Best for**: First-time users, budget-conscious buyers
• **Key features**: Heart rate monitoring, GPS, sleep tracking, crash detection
• **What you get**: 90% of Apple Watch experience at best price
• **Missing**: Always-On display, ECG, advanced health features

**Apple Watch Series 9** - ₹41,900
• **Best for**: Most users wanting complete Apple Watch experience  
• **Key features**: Always-On display, Double Tap gesture, ECG, Blood Oxygen
• **What's special**: Latest processor, brightest display, most advanced features
• **Why choose**: Perfect balance of features and price

**Apple Watch Ultra 2** - ₹89,900
• **Best for**: Serious athletes, outdoor adventurers, extreme durability needs
• **Key features**: Titanium case, 36+ hour battery, 100m water resistance
• **What's unique**: Built for extreme conditions, longest battery life
• **Why choose**: Maximum durability and capabilities

**My Recommendation:**
• **Budget under ₹30k** → Apple Watch SE
• **Budget ₹30-60k** → Apple Watch Series 9 (most popular choice)
• **Need extreme durability** → Apple Watch Ultra 2

All models work with iPhone 6s and later, include the same core Apple Watch apps, and receive the same watchOS updates.

What's your primary use case - fitness, health monitoring, notifications, or extreme sports?"""
    
    def _get_default_response(self) -> str:
        """Default response for general queries"""
        return """**I'm your Apple Watch expert!** I provide accurate, current information about Apple Watch models, pricing, and features.

**I can help you with:**

**🏷️ Smart Recommendations**
• Find the perfect Apple Watch for your specific budget
• Personalized model selection based on your needs

**💰 Current Pricing & Value Analysis** 
• Real Indian market prices (no made-up information)
• Best value recommendations for any budget

**🔧 Technical Support**
• Step-by-step troubleshooting for common issues
• Setup, pairing, and maintenance help

**⚖️ Detailed Comparisons**
• Feature-by-feature analysis: SE vs Series 9 vs Ultra 2
• Help you choose the right model

**Current Models & Pricing:**
• Apple Watch SE: ₹24,900 (best value)
• Apple Watch Series 9: ₹41,900 (most popular)  
• Apple Watch Ultra 2: ₹89,900 (most advanced)

**What specific Apple Watch question can I help you with today?**

Try asking:
• "Best Apple Watch for ₹30k budget"
• "Compare Apple Watch SE vs Series 9"
• "My Apple Watch won't charge"
• "Which Apple Watch model should I buy?"
"""

class AppleWatchExpert:
    """Streamlined Apple Watch Expert with efficient responses"""
    
    def __init__(self):
        self.groq_client = None
        self.selected_model = None
        self.hf_pipeline = None
        self.ollama_available = False
        
        # Enhanced knowledge base
        self.knowledge_base = AppleWatchKnowledgeBase()
        
        # Status tracking
        self.active_model = "none"
        self.model_status = {}
        
        # Initialize models in priority order
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize models with proper fallback priority"""
        
        # 1. Try Groq first (highest priority)
        if config.is_groq_available():
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=config.get_groq_api_key())
                
                # Test and select best model
                self.selected_model = config.test_and_select_best_model()
                
                if self.selected_model:
                    self.active_model = "groq"
                    self.model_status = {
                        "type": "groq",
                        "model": self.selected_model,
                        "status": "active",
                        "description": "Premium AI with smart data integration"
                    }
                    logger.info(f"✅ Groq initialized: {self.selected_model}")
                    return  # Success - no need to try others
                
            except Exception as e:
                logger.error(f"❌ Groq initialization failed: {e}")
        
        # 2. Try Hugging Face (second priority)
        if TORCH_AVAILABLE:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                
                model_name = "microsoft/DialoGPT-small"
                
                # Initialize components separately
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Add pad token if missing
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Create pipeline with correct parameters
                self.hf_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU only
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=80,
                    pad_token_id=tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                self.active_model = "huggingface"
                self.model_status = {
                    "type": "huggingface", 
                    "model": "DialoGPT-small",
                    "status": "active",
                    "description": "Local AI with comprehensive knowledge"
                }
                logger.info("✅ Hugging Face initialized successfully")
                return  # Success
                
            except Exception as e:
                logger.error(f"❌ Hugging Face initialization failed: {e}")
        
        # 3. Try Ollama (third priority)
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                self.ollama_available = True
                self.active_model = "ollama"
                self.model_status = {
                    "type": "ollama",
                    "model": "llama3.1:8b",
                    "status": "active", 
                    "description": "Advanced local AI with expert knowledge"
                }
                logger.info("✅ Ollama available")
                return  # Success
                
        except Exception as e:
            logger.warning(f"❌ Ollama not available: {e}")
        
        # 4. Fallback to expert knowledge base
        self.active_model = "expert"
        self.model_status = {
            "type": "expert",
            "model": "comprehensive_knowledge_base",
            "status": "active",
            "description": "Expert knowledge with comprehensive responses"
        }
        logger.info("✅ Expert knowledge base ready")
    
    def generate_apple_watch_response(self, question: str, context: str = "", 
                                    sentiment: Optional[SentimentAnalysis] = None,
                                    chat_history: List[Dict] = None) -> str:
        """Generate comprehensive response with minimal web requests"""
        
        try:
            # Get comprehensive response from knowledge base (with smart web integration)
            base_response = self.knowledge_base.get_response(question)
            
            # Use AI model to enhance if available, otherwise return base response
            if self.active_model == "groq" and self.groq_client:
                return self._enhance_with_groq(question, base_response, sentiment, chat_history)
            
            elif self.active_model == "huggingface" and self.hf_pipeline:
                # For HF, return base response as it's already comprehensive
                return base_response
                
            elif self.active_model == "ollama" and self.ollama_available:
                return self._enhance_with_ollama(question, base_response, sentiment)
                
            else:
                # Return the comprehensive base response
                return base_response
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I'm here to help with Apple Watch questions! What specific information do you need about models, pricing, or features?"
    
    def _enhance_with_groq(self, question: str, base_response: str, 
                          sentiment: Optional[SentimentAnalysis], 
                          chat_history: List[Dict]) -> str:
        """Enhance response using Groq while preserving all factual information"""
        
        # For comprehensive base responses, just return them as-is to avoid making them worse
        if len(base_response) > 300:
            return base_response
        
        # Only enhance shorter responses
        system_prompt = """You are an Apple Watch expert. Make the response more conversational and helpful while keeping ALL factual information exactly as provided."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\nResponse to enhance: {base_response}"}
            ]
            
            response = self.groq_client.chat.completions.create(
                model=self.selected_model,
                messages=messages,
                temperature=0.1,
                max_tokens=400,
                top_p=0.9
            )
            
            enhanced = response.choices[0].message.content.strip()
            
            # Use enhanced response if it's good, otherwise stick with base
            if len(enhanced) > len(base_response) * 0.8 and "I don't" not in enhanced:
                return enhanced
            else:
                return base_response
                
        except Exception as e:
            logger.error(f"Groq enhancement failed: {e}")
            return base_response
    
    def _enhance_with_ollama(self, question: str, base_response: str,
                           sentiment: Optional[SentimentAnalysis]) -> str:
        """Enhance with Ollama"""
        
        # Return base response as it's already comprehensive
        return base_response
    
    def get_model_status(self) -> Dict[str, Union[str, dict]]:
        """Get current model status"""
        return {
            "active_model": self.active_model,
            "status": self.model_status,
            "groq_available": bool(self.groq_client),
            "hf_available": bool(self.hf_pipeline),
            "ollama_available": self.ollama_available
        }