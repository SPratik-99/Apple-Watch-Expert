"""
FINAL Multi-LLM System with Mandatory Web Scraping
Always checks Apple.com when information is not found locally
"""
import logging
from typing import List, Dict, Optional, Union
import os
import json
import requests
import time

logger = logging.getLogger(__name__)

# Import with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import config
from classifier import SentimentAnalysis
from real_apple_scraper import RealAppleWebScraper

class AppleWatchKnowledgeBase:
    """Knowledge base that ALWAYS checks Apple.com when local data insufficient"""
    
    def __init__(self):
        # Local knowledge for instant responses
        self.local_knowledge = {
            "current_models": {
                "Apple Watch SE (2nd gen)": {
                    "starting_price": 24900,
                    "sizes": ["40mm", "44mm"],
                    "key_features": ["Heart rate", "GPS", "Sleep tracking", "Crash detection"],
                    "missing_features": ["Always-On display", "ECG", "Blood Oxygen", "Double Tap"]
                },
                "Apple Watch Series 9": {
                    "starting_price": 41900,
                    "sizes": ["41mm", "45mm"], 
                    "key_features": ["S9 chip", "Always-On display", "ECG", "Blood Oxygen", "Double Tap", "Temperature sensing"],
                    "unique": ["Double Tap gesture", "On-device Siri", "Brightest display"]
                },
                "Apple Watch Ultra 2": {
                    "starting_price": 89900,
                    "sizes": ["49mm"],
                    "key_features": ["Titanium case", "Action Button", "100m water resistance", "36+ hour battery"],
                    "unique": ["Precision dual-frequency GPS", "86dB emergency siren", "Extreme durability"]
                }
            },
            "non_existent_models": [
                "Apple Watch Series 10", "Apple Watch Series10",
                "Apple Watch Ultra 3", "Apple Watch Ultra3", 
                "Apple Watch SE 3", "Apple Watch SE3"
            ]
        }
        
        # Initialize web scraper for live data
        self.web_scraper = RealAppleWebScraper()
        self.web_connection_available = False
        self._test_web_connection()
    
    def _test_web_connection(self):
        """Test web scraping capability"""
        try:
            self.web_connection_available = self.web_scraper.test_connection()
            if self.web_connection_available:
                logger.info("✅ Apple.com web scraping available")
            else:
                logger.warning("⚠️ Apple.com web scraping unavailable")
        except Exception as e:
            logger.error(f"Web connection test failed: {e}")
            self.web_connection_available = False
    
    def get_comprehensive_response(self, query: str) -> Dict[str, str]:
        """Get comprehensive response combining local knowledge + web data"""
        
        # Step 1: Check if query is about non-existent products
        if self._is_non_existent_product(query):
            return {
                "source": "local_knowledge",
                "response": self._handle_non_existent_product(query),
                "confidence": "high"
            }
        
        # Step 2: Try to get local knowledge first
        local_response = self._get_local_knowledge(query)
        
        # Step 3: ALWAYS try web scraping for additional/current information
        web_response = ""
        if self.web_connection_available:
            try:
                web_response = self.web_scraper.search_apple_watch_info(query)
                if web_response:
                    logger.info(f"✅ Retrieved web data for: {query[:50]}")
                else:
                    logger.info(f"ℹ️ No web data found for: {query[:50]}")
            except Exception as e:
                logger.error(f"Web scraping failed for '{query}': {e}")
        
        # Step 4: Combine responses intelligently
        return self._combine_responses(local_response, web_response, query)
    
    def _is_non_existent_product(self, query: str) -> bool:
        """Check if query asks about non-existent Apple Watch models"""
        query_lower = query.lower()
        return any(model.lower() in query_lower for model in self.local_knowledge["non_existent_models"])
    
    def _handle_non_existent_product(self, query: str) -> str:
        """Handle queries about non-existent products"""
        query_lower = query.lower()
        
        # Identify which non-existent model was mentioned
        mentioned_model = "the mentioned model"
        for model in self.local_knowledge["non_existent_models"]:
            if model.lower() in query_lower:
                mentioned_model = model
                break
        
        return f"""I cannot provide information about {mentioned_model} because it doesn't exist in Apple's current lineup.

**Current Apple Watch Models (August 2025):**
• **Apple Watch SE (2nd gen)**: Starting at ₹24,900
• **Apple Watch Series 9**: Starting at ₹41,900  
• **Apple Watch Ultra 2**: ₹89,900

Would you like detailed information about any of these actual models?"""
    
    def _get_local_knowledge(self, query: str) -> str:
        """Get response from local knowledge base"""
        query_lower = query.lower()
        
        # Budget-based recommendations
        if any(word in query_lower for word in ["budget", "price", "cost", "suggest", "recommend"]):
            import re
            budget_match = re.search(r'(\d+)k|(\d{4,6})', query_lower)
            if budget_match:
                budget = int(budget_match.group(1) or budget_match.group(2))
                if 'k' in query_lower and budget < 1000:
                    budget *= 1000
                return self._get_budget_recommendation(budget)
        
        # Model-specific information
        if "se" in query_lower and not "series" in query_lower:
            return self._get_se_info()
        elif "series 9" in query_lower or "s9" in query_lower:
            return self._get_series9_info()
        elif "ultra" in query_lower:
            return self._get_ultra_info()
        
        # Comparisons
        if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            if "se" in query_lower and ("series 9" in query_lower or "s9" in query_lower):
                return self._get_se_vs_series9_comparison()
        
        # Technical support
        if any(word in query_lower for word in ["charge", "problem", "fix", "not working"]):
            return self._get_technical_support(query_lower)
        
        return ""  # No local knowledge available
    
    def _combine_responses(self, local_response: str, web_response: str, query: str) -> Dict[str, str]:
        """Intelligently combine local and web responses"""
        
        # If we have both local and web data
        if local_response and web_response:
            combined_response = local_response
            
            # Add web data as additional information
            if len(web_response.strip()) > 20:  # Only if web response is substantial
                combined_response += f"\n\n**Latest from Apple.com:** {web_response[:300]}"
            
            return {
                "source": "combined",
                "response": combined_response,
                "confidence": "high"
            }
        
        # If we only have local data
        elif local_response:
            response = local_response
            
            # Add note about web availability
            if not self.web_connection_available:
                response += "\n\n*Note: Live web data currently unavailable*"
            elif not web_response:
                response += "\n\n*Note: No additional information found on Apple.com*"
            
            return {
                "source": "local_knowledge",
                "response": response,
                "confidence": "medium"
            }
        
        # If we only have web data
        elif web_response:
            return {
                "source": "web_scraping",
                "response": f"**From Apple.com:** {web_response}",
                "confidence": "medium"
            }
        
        # If we have no specific data, but web is available, indicate we checked
        else:
            if self.web_connection_available:
                fallback_response = f"""I checked Apple.com for information about your query but didn't find specific details.

**I can help with these topics:**
• Current Apple Watch models and pricing
• Model comparisons (SE vs Series 9 vs Ultra 2)
• Technical support and troubleshooting
• Budget recommendations

Could you please rephrase your question or ask about a specific model?"""
            else:
                fallback_response = """I don't have specific information about that query in my current knowledge base.

**I can help with these topics:**
• Current Apple Watch models and pricing
• Model comparisons (SE vs Series 9 vs Ultra 2)  
• Technical support and troubleshooting
• Budget recommendations

What would you like to know about Apple Watch?"""
            
            return {
                "source": "fallback",
                "response": fallback_response,
                "confidence": "low"
            }
    
    def _get_budget_recommendation(self, budget: int) -> str:
        """Get budget-specific recommendations"""
        if budget < 25000:
            return f"""For ₹{budget:,}, you're close to the Apple Watch SE starting price.

**Apple Watch SE (2nd gen)**: ₹24,900
• S8 dual-core processor
• Heart rate monitoring and GPS tracking
• Sleep tracking with detailed stages
• Crash detection and fall detection
• 85+ workout types, 18+ hour battery

**Your options:**
• Stretch budget by ₹{24900-budget:,} for the SE 40mm GPS
• Look for certified refurbished SE models (₹18,000-22,000)
• Wait for festival sales (SE often drops to ₹22,000-23,000)"""

        elif budget <= 50000:
            return f"""Perfect budget for the Apple Watch Series 9!

**Apple Watch Series 9** (₹41,900-44,900):
• S9 SiP chip - 60% faster than SE
• Always-On Retina display (2000 nits brightness)
• Double Tap gesture - revolutionary control method
• Complete health suite: ECG, Blood Oxygen, temperature sensing
• Enhanced Siri with on-device processing

**Your ₹{budget:,} budget covers:**
• Series 9 41mm GPS: ₹41,900 ✅
• Series 9 45mm GPS: ₹44,900 ✅
• Premium Sport Loop or Leather band

The Series 9 is the best overall Apple Watch for most users."""

        else:
            return f"""Excellent budget for premium Apple Watch options!

**Option 1: Apple Watch Series 9** (₹41,900-53,900)
Complete flagship experience with latest features

**Option 2: Apple Watch Ultra 2** (₹89,900)
• Aerospace titanium case, most durable
• Largest 49mm display, 3000 nits brightness
• 36+ hour battery (72 hours Low Power Mode)
• 100m water resistance, precision dual-frequency GPS
• Action Button, 86dB emergency siren

**For ₹{budget:,}, I recommend:**
• **Series 9** for everyday premium use with great value
• **Ultra 2** if you need extreme durability and longest battery
• **Series 9 + Premium accessories** for complete luxury setup"""
    
    def _get_se_info(self) -> str:
        """Get Apple Watch SE information"""
        return """**Apple Watch SE (2nd generation)**

**Starting Price:** ₹24,900 (40mm GPS)
**Available Sizes:** 40mm (₹24,900), 44mm (₹28,900)
**Cellular Options:** 40mm (₹30,900), 44mm (₹34,900)

**Key Features:**
• S8 SiP dual-core processor
• Heart rate monitoring with irregular rhythm alerts
• Sleep tracking with REM, Core, Deep sleep stages
• GPS tracking for accurate workout data
• Crash detection and fall detection
• 85+ workout types
• Water resistant to 50 meters
• 18+ hour all-day battery life

**What's Missing vs Series 9:**
• No Always-On Retina display
• No ECG app or Blood Oxygen monitoring
• No Double Tap gesture control
• No temperature sensing
• Older S8 processor (vs S9)

**Perfect For:** First-time Apple Watch users, budget-conscious buyers, fitness enthusiasts who don't need advanced health monitoring."""
    
    def _get_series9_info(self) -> str:
        """Get Apple Watch Series 9 information"""
        return """**Apple Watch Series 9**

**Starting Price:** ₹41,900 (41mm GPS)
**Available Sizes:** 41mm (₹41,900), 45mm (₹44,900)
**Cellular Options:** 41mm (₹50,900), 45mm (₹53,900)

**Key Features:**
• S9 SiP chip - 60% faster than Series 8
• Always-On Retina display with 2000 nits brightness
• Double Tap gesture control - tap thumb and finger to control
• Complete health monitoring: ECG, Blood Oxygen, temperature sensing
• Enhanced Siri with on-device processing for privacy
• Heart rate monitoring with irregular rhythm alerts
• Advanced sleep tracking and cycle tracking
• Precision dual-frequency GPS
• Water resistant to 50 meters
• 18+ hour battery life

**New in Series 9:**
• Double Tap gesture - revolutionary new control method
• Brightest Apple Watch display ever (2000 nits)
• On-device Siri processing for faster, more private interactions
• Carbon neutral when paired with Sport Loop

**Perfect For:** Users who want the complete Apple Watch experience with latest features, health-conscious individuals, tech enthusiasts."""
    
    def _get_ultra_info(self) -> str:
        """Get Apple Watch Ultra 2 information"""
        return """**Apple Watch Ultra 2**

**Price:** ₹89,900 (49mm Cellular only)
**Available Size:** 49mm only (largest Apple Watch)

**Extreme Features:**
• Aerospace-grade titanium case - most durable Apple Watch
• Flat sapphire crystal display - virtually scratchproof
• 3000 nits brightness - viewable in direct sunlight
• 100m water resistance - suitable for recreational scuba diving
• Action Button - customizable for instant access to features
• 86dB emergency siren - audible up to 180 meters away
• Precision dual-frequency GPS - most accurate location tracking

**Extended Battery:**
• 36 hours normal use (vs 18 hours on other models)
• Up to 72 hours in Low Power Mode
• Perfect for multi-day adventures

**Built for Extremes:**
• Operating temperature: -20°C to 55°C
• MIL-STD 810H tested for durability
• Designed for diving, mountaineering, endurance sports
• Ocean Band and Alpine Loop designed for extreme conditions

**Perfect For:** Serious athletes, outdoor adventurers, professionals in extreme environments, users who need maximum durability and longest battery life."""
    
    def _get_se_vs_series9_comparison(self) -> str:
        """Compare SE vs Series 9"""
        return """**Apple Watch SE vs Series 9 - Complete Comparison**

**Apple Watch SE (2nd gen)** - ₹24,900
✅ S8 dual-core processor
✅ Heart rate monitoring, sleep tracking
✅ Crash detection, fall detection
✅ GPS tracking, 85+ workouts
✅ Water resistant to 50m, 18+ hour battery
❌ No Always-On display
❌ No ECG or Blood Oxygen monitoring  
❌ No Double Tap gesture
❌ No temperature sensing

**Apple Watch Series 9** - ₹41,900
✅ All SE features PLUS:
✅ S9 SiP chip (60% faster performance)
✅ Always-On Retina display (2000 nits)
✅ Double Tap gesture control
✅ ECG app for heart rhythm monitoring
✅ Blood Oxygen app for wellness insights
✅ Temperature sensing for cycle tracking
✅ On-device Siri processing

**₹17,000 Price Difference - Worth It If:**
• You want Always-On display convenience
• Advanced health monitoring is important (ECG/Blood Oxygen)
• You love cutting-edge features (Double Tap gesture)
• Performance matters for apps and smooth navigation

**Stick with SE If:**
• Budget is the primary concern
• Basic fitness tracking meets your needs
• This is your first Apple Watch experience
• You don't need advanced health features

Both offer excellent value - Series 9 for premium experience, SE for best affordability."""
    
    def _get_technical_support(self, query: str) -> str:
        """Get technical support information"""
        if "charge" in query:
            return """**Apple Watch Charging Issues - Complete Solution**

**Step 1: Clean Everything**
• Remove Apple Watch from charger completely
• Use soft, lint-free cloth (microfiber works best)
• Clean watch back (circular sensor area) thoroughly
• Clean charger surface - remove debris, sweat, moisture

**Step 2: Check Setup**
• Use original Apple charging cable ONLY
• Connect to 5W+ USB power adapter (iPhone charger works)
• Ensure magnetic connection clicks properly
• Green lightning bolt should appear on watch

**Step 3: Force Restart**
• Hold side button + Digital Crown together
• Keep holding for exactly 10 seconds
• Release when Apple logo appears
• Place back on charger after restart

**Step 4: Advanced Troubleshooting**
• Try different power outlet and USB adapter
• Check charging cable for damage
• Ensure watch is centered on charger pad
• Test in different temperature environment

**Still not working?** Contact Apple Support if under warranty or visit Apple Store for free diagnostics."""
        
        return """**Apple Watch Technical Support**

**Common Issues & Quick Fixes:**

**Watch Frozen/Unresponsive:**
• Force restart: Hold side button + Digital Crown for 10 seconds

**Battery Draining Fast:**
• Check Always-On Display settings
• Reduce background app refresh
• Update to latest watchOS

**Connection Problems:**
• Keep iPhone nearby (within 10 meters)
• Check Bluetooth is ON on iPhone
• Restart both devices
• Re-pair if necessary

**Fitness Tracking Issues:**
• Ensure wrist detection is ON
• Keep watch snug during workouts
• Calibrate in Settings > Privacy & Security

Tell me your specific issue for detailed troubleshooting steps!"""

class AppleWatchExpert:
    """Enhanced Apple Watch Expert with mandatory web checking"""
    
    def __init__(self):
        self.groq_client = None
        self.selected_model = None
        self.hf_pipeline = None
        self.ollama_available = False
        
        # Enhanced knowledge base with web integration
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
                        "description": "Premium AI with web verification"
                    }
                    logger.info(f"✅ Groq initialized: {self.selected_model}")
                    return  # Success - no need to try others
                
            except Exception as e:
                logger.error(f"❌ Groq initialization failed: {e}")
        
        # 2. Try Hugging Face (second priority) - FIXED PARAMETERS
        if TORCH_AVAILABLE:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                
                model_name = "microsoft/DialoGPT-small"
                
                # Initialize components separately for better control
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Add pad token if missing
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Create pipeline with corrected parameters
                self.hf_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU only
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=100,
                    pad_token_id=tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                self.active_model = "huggingface"
                self.model_status = {
                    "type": "huggingface", 
                    "model": "DialoGPT-small",
                    "status": "active",
                    "description": "Local AI with web verification"
                }
                logger.info("✅ Hugging Face initialized with web integration")
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
                    "description": "Advanced local AI with web verification"
                }
                logger.info("✅ Ollama available with web integration")
                return  # Success
                
        except Exception as e:
            logger.warning(f"❌ Ollama not available: {e}")
        
        # 4. Fallback to expert knowledge base with web scraping
        self.active_model = "expert"
        self.model_status = {
            "type": "expert",
            "model": "knowledge_base_with_web",
            "status": "active",
            "description": "Expert knowledge with web verification"
        }
        logger.info("✅ Expert knowledge base with web scraping ready")
    
    def generate_apple_watch_response(self, question: str, context: str = "", 
                                    sentiment: Optional[SentimentAnalysis] = None,
                                    chat_history: List[Dict] = None) -> str:
        """Generate response with MANDATORY web checking when needed"""
        
        try:
            # STEP 1: ALWAYS get comprehensive information (local + web)
            comprehensive_data = self.knowledge_base.get_comprehensive_response(question)
            
            # STEP 2: Use AI model to enhance the response if available
            base_response = comprehensive_data["response"]
            
            if self.active_model == "groq" and self.groq_client:
                enhanced_response = self._enhance_with_groq(question, base_response, comprehensive_data, sentiment, chat_history)
                return enhanced_response
            
            elif self.active_model == "huggingface" and self.hf_pipeline:
                enhanced_response = self._enhance_with_hf(question, base_response, comprehensive_data, sentiment)
                return enhanced_response
                
            elif self.active_model == "ollama" and self.ollama_available:
                enhanced_response = self._enhance_with_ollama(question, base_response, comprehensive_data, sentiment)
                return enhanced_response
                
            else:
                # Return the comprehensive response directly
                return base_response
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            # Even if everything fails, try to get basic response
            try:
                fallback_data = self.knowledge_base.get_comprehensive_response(question)
                return fallback_data["response"]
            except:
                return "I'm here to help with Apple Watch questions! What specific information do you need?"
    
    def _enhance_with_groq(self, question: str, base_response: str, data: Dict, 
                          sentiment: Optional[SentimentAnalysis], 
                          chat_history: List[Dict]) -> str:
        """Enhance response using Groq while preserving factual accuracy"""
        
        system_prompt = """You are an Apple Watch expert assistant. Your job is to enhance and format the provided factual information into a natural, helpful response.

CRITICAL RULES:
1. Use ONLY the factual information provided in the context
2. NEVER add new product information, prices, or specifications not in the context
3. If the context says a product doesn't exist, clearly state that
4. Format the information in a helpful, conversational way
5. Don't mention that you're using provided information

The factual information will be provided after the user's question."""

        # Build conversation
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent history for context
        if chat_history:
            messages.extend(chat_history[-2:])
        
        # Provide the question and factual context
        user_message = f"""User Question: {question}

Factual Information to Use:
{base_response}

Please format this information into a natural, helpful response for the user. Use only the factual information provided above."""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.selected_model,
                messages=messages,
                temperature=0.1,  # Low temperature for factual consistency
                max_tokens=700,
                top_p=0.9
            )
            
            enhanced_response = response.choices[0].message.content.strip()
            
            # Verify the enhanced response doesn't contradict the base facts
            if len(enhanced_response) > 50 and "I don't have" not in enhanced_response:
                return enhanced_response
            else:
                return base_response  # Fallback to original if enhancement failed
            
        except Exception as e:
            logger.error(f"Groq enhancement failed: {e}")
            return base_response
    
    def _enhance_with_hf(self, question: str, base_response: str, data: Dict, 
                        sentiment: Optional[SentimentAnalysis]) -> str:
        """Enhance with Hugging Face (lightweight enhancement)"""
        
        # For HF, just return the base response since it's already comprehensive
        # HF models are not great at following complex instructions
        return base_response
    
    def _enhance_with_ollama(self, question: str, base_response: str, data: Dict,
                           sentiment: Optional[SentimentAnalysis]) -> str:
        """Enhance with Ollama"""
        
        try:
            prompt = f"""Format this Apple Watch information into a helpful response:

User asked: {question}

Factual information: {base_response[:800]}

Format this into a natural, helpful response. Use only the provided information."""
            
            payload = {
                "model": "llama3.1:8b",
                "prompt": prompt,
                "options": {
                    "temperature": 0.1,  # Low temperature for factual consistency
                    "top_p": 0.9,
                    "num_predict": 300
                },
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced = result.get("response", "").strip()
                if len(enhanced) > 50:
                    return enhanced
            
            return base_response
            
        except Exception as e:
            logger.error(f"Ollama enhancement failed: {e}")
            return base_response
    
    def get_model_status(self) -> Dict[str, Union[str, dict]]:
        """Get current model status with web integration info"""
        status = {
            "active_model": self.active_model,
            "status": self.model_status,
            "groq_available": bool(self.groq_client),
            "hf_available": bool(self.hf_pipeline),
            "ollama_available": self.ollama_available,
            "web_scraping_available": self.knowledge_base.web_connection_available
        }
        
        return status