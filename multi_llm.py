"""
FIXED: Mandatory Apple.com Web Scraping When Local Data Missing
Always checks Apple website when PDFs/JSON/TXT files don't have the answer
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
from apple_search import RealAppleWebScraper

class AppleWatchKnowledgeBase:
    """Knowledge base that ALWAYS checks Apple.com when local data is missing"""
    
    def __init__(self):
        # Initialize web scraper FIRST
        self.web_scraper = RealAppleWebScraper()
        self.web_available = False
        
        # Test web connection immediately
        self._test_web_connection()
        
        # Minimal local knowledge - forces web scraping for most queries
        self.basic_knowledge = {
            "models_exist": ["SE", "Series 9", "Ultra 2"],
            "models_dont_exist": ["Series 10", "Series10", "Ultra 3", "SE 3"],
            "basic_pricing": {
                "SE": "₹24,900+",
                "Series 9": "₹41,900+", 
                "Ultra 2": "₹89,900"
            }
        }
        
        # Cache for web responses (short duration to stay current)
        self.web_cache = {}
        self.cache_duration = 600  # 10 minutes only
    
    def _test_web_connection(self):
        """Test and establish web scraping capability"""
        try:
            self.web_available = self.web_scraper.test_connection()
            if self.web_available:
                logger.info("✅ Apple.com web scraping ACTIVE - will check for current data")
            else:
                logger.warning("⚠️ Apple.com web scraping FAILED - using fallback knowledge only")
        except Exception as e:
            logger.error(f"Web connection test failed: {e}")
            self.web_available = False
    
    def get_comprehensive_response(self, query: str) -> Dict[str, str]:
        """Get response - ALWAYS tries web scraping first when local data insufficient"""
        
        query_lower = query.lower()
        
        # Step 1: Handle obviously non-existent products locally (no web needed)
        if self._is_non_existent_product(query_lower):
            return {
                "source": "local_definitive",
                "response": self._handle_non_existent_product(query),
                "web_checked": False
            }
        
        # Step 2: For ALL other queries, try web scraping FIRST
        web_response = ""
        web_checked = False
        
        if self.web_available:
            try:
                logger.info(f"🌐 Checking Apple.com for: {query[:50]}...")
                web_response = self._get_web_data(query)
                web_checked = True
                
                if web_response and len(web_response.strip()) > 50:
                    logger.info(f"✅ Found current data on Apple.com for: {query[:30]}")
                    
                    # If web data is comprehensive, use it primarily
                    return {
                        "source": "web_primary",
                        "response": self._format_web_response(web_response, query),
                        "web_checked": True
                    }
                else:
                    logger.info(f"ℹ️ Limited data on Apple.com for: {query[:30]}")
                    
            except Exception as e:
                logger.error(f"Web scraping failed for '{query}': {e}")
        
        # Step 3: Fall back to local knowledge + indicate web was checked
        local_response = self._get_local_response(query_lower)
        
        if local_response:
            # Combine local knowledge with web status
            response = local_response
            if web_checked and not web_response:
                response += f"\n\n*Note: Checked Apple.com for latest information - using verified knowledge base.*"
            elif not web_checked:
                response += f"\n\n*Note: Apple.com currently unavailable - using verified knowledge base.*"
                
            return {
                "source": "local_enhanced",
                "response": response,
                "web_checked": web_checked
            }
        
        # Step 4: If no local knowledge and no web data, be explicit
        if web_checked:
            fallback = f"""I checked Apple.com for information about your query but couldn't find specific details.

**I can provide current information about:**
• Apple Watch SE (₹24,900+) - Most affordable option
• Apple Watch Series 9 (₹41,900+) - Latest mainstream model  
• Apple Watch Ultra 2 (₹89,900) - Most advanced model

**For specific questions, try:**
• "Apple Watch SE details and pricing"
• "Compare Series 9 vs SE" 
• "Apple Watch Ultra 2 features"
• "Best Apple Watch for ₹30k budget"

What specific Apple Watch information can I help you find?"""
        else:
            fallback = f"""I don't have specific data about that query, and Apple.com is currently unavailable.

**I can help with these verified topics:**
• Current Apple Watch models and basic pricing
• General feature comparisons
• Common troubleshooting issues

Could you please ask about a specific model or feature?"""
        
        return {
            "source": "fallback",
            "response": fallback,
            "web_checked": web_checked
        }
    
    def _get_web_data(self, query: str) -> str:
        """Get data from Apple website with caching"""
        query_key = query.lower()[:50]  # Cache key
        
        # Check cache first
        if query_key in self.web_cache:
            cache_time, cached_data = self.web_cache[query_key]
            if time.time() - cache_time < self.cache_duration:
                logger.info(f"Using cached Apple.com data for: {query[:30]}")
                return cached_data
        
        # Fetch from web
        try:
            web_data = self.web_scraper.search_apple_watch_info(query)
            
            # Cache the result
            if web_data:
                self.web_cache[query_key] = (time.time(), web_data)
                logger.info(f"Cached new Apple.com data for: {query[:30]}")
            
            return web_data or ""
            
        except Exception as e:
            logger.error(f"Web data fetch failed: {e}")
            return ""
    
    def _format_web_response(self, web_data: str, query: str) -> str:
        """Format web response with context"""
        query_lower = query.lower()
        
        # Add context based on query type
        if any(word in query_lower for word in ["price", "cost", "budget"]):
            formatted = f"**Current Apple Watch Information from Apple.com:**\n\n{web_data}"
            
            # Add local context for budget queries
            if any(word in query_lower for word in ["budget", "suggest", "recommend"]):
                budget = self._extract_budget(query)
                if budget:
                    formatted += f"\n\n**For your ₹{budget:,} budget:**\n{self._get_budget_context(budget)}"
                    
        elif any(word in query_lower for word in ["compare", "vs", "difference"]):
            formatted = f"**Apple Watch Comparison from Apple.com:**\n\n{web_data}"
            
        elif any(word in query_lower for word in ["features", "specs", "what"]):
            formatted = f"**Apple Watch Features from Apple.com:**\n\n{web_data}"
            
        else:
            formatted = f"**From Apple.com:**\n\n{web_data}"
        
        # Add timestamp for freshness
        formatted += f"\n\n*Information current as of {time.strftime('%B %d, %Y')}*"
        
        return formatted
    
    def _is_non_existent_product(self, query: str) -> bool:
        """Check if query is about non-existent products"""
        return any(model.lower() in query for model in self.basic_knowledge["models_dont_exist"])
    
    def _handle_non_existent_product(self, query: str) -> str:
        """Handle non-existent product queries"""
        mentioned_model = "that model"
        for model in self.basic_knowledge["models_dont_exist"]:
            if model.lower() in query.lower():
                mentioned_model = f"Apple Watch {model}"
                break
        
        return f"""I cannot provide information about {mentioned_model} because it doesn't exist in Apple's current lineup.

**Current Apple Watch Models (Verified August 2025):**

**Apple Watch SE (2nd gen)** - Starting at ₹24,900
• Most affordable genuine Apple Watch
• Heart rate monitoring, GPS, sleep tracking, crash detection

**Apple Watch Series 9** - Starting at ₹41,900  
• Latest flagship model with Always-On display
• Complete health suite: ECG, Blood Oxygen, temperature sensing
• Double Tap gesture control

**Apple Watch Ultra 2** - ₹89,900
• Most advanced and durable Apple Watch
• Titanium case, 36+ hour battery, 100m water resistance
• Built for extreme sports and adventures

Which of these actual models would you like detailed information about?"""
    
    def _get_local_response(self, query: str) -> str:
        """Get response from minimal local knowledge"""
        
        # Budget recommendations (always try web first, this is fallback)
        if any(word in query for word in ["budget", "price", "suggest", "recommend"]):
            budget = self._extract_budget(query)
            if budget:
                return self._get_budget_recommendation(budget)
        
        # Basic model info (minimal - encourages web checking)
        if "se" in query and "series" not in query:
            return f"""**Apple Watch SE (2nd generation)**
Starting at {self.basic_knowledge["basic_pricing"]["SE"]}

*For complete current specifications, pricing, and availability, I recommend checking Apple.com directly or asking for specific features you're interested in.*"""
        
        elif "series 9" in query or "s9" in query:
            return f"""**Apple Watch Series 9**
Starting at {self.basic_knowledge["basic_pricing"]["Series 9"]}

*For complete current specifications, pricing, and availability, I recommend checking Apple.com directly or asking for specific features you're interested in.*"""
        
        elif "ultra" in query:
            return f"""**Apple Watch Ultra 2**
Price: {self.basic_knowledge["basic_pricing"]["Ultra 2"]}

*For complete current specifications, pricing, and availability, I recommend checking Apple.com directly or asking for specific features you're interested in.*"""
        
        # General queries
        if any(word in query for word in ["model", "which", "what", "available"]):
            return f"""**Current Apple Watch Models:**
• Apple Watch SE: {self.basic_knowledge["basic_pricing"]["SE"]}
• Apple Watch Series 9: {self.basic_knowledge["basic_pricing"]["Series 9"]}
• Apple Watch Ultra 2: {self.basic_knowledge["basic_pricing"]["Ultra 2"]}

*For detailed specifications, current availability, and exact pricing, please ask about a specific model or check Apple.com.*"""
        
        return ""  # No local knowledge - forces comprehensive response above
    
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
    
    def _get_budget_recommendation(self, budget: int) -> str:
        """Get budget recommendation - enhanced with web data note"""
        if budget < 25000:
            return f"""For ₹{budget:,} budget, you're very close to the Apple Watch SE.

**Apple Watch SE (2nd gen)** - Starting at ₹24,900
• Most affordable genuine Apple Watch
• Essential features: Heart rate, GPS, sleep tracking, crash detection
• Missing: Always-On display, ECG, advanced health monitoring

**Options for your budget:**
• Stretch by ₹{24900-budget:,} for SE 40mm GPS
• Look for certified refurbished SE (₹18,000-22,000 range)
• Wait for festival sales (SE can drop to ₹22,000-23,000)

*For current exact pricing and availability, check Apple.com or ask me to look up specific current offers.*"""

        elif budget <= 50000:
            return f"""Perfect! ₹{budget:,} is ideal for Apple Watch Series 9.

**Apple Watch Series 9** - Starting at ₹41,900
• Latest flagship with S9 chip (60% faster than SE)
• Always-On Retina display, Double Tap gesture
• Complete health suite: ECG, Blood Oxygen, temperature sensing

**Your budget covers:**
• Series 9 41mm GPS: ₹41,900 ✅
• Series 9 45mm GPS: ₹44,900 ✅  
• Premium bands with remaining budget

*For current exact pricing, cellular options, and availability, I can check Apple.com for you.*"""

        else:
            return f"""Excellent ₹{budget:,} budget for premium options!

**Apple Watch Series 9** (₹41,900-53,900)
• Complete flagship experience

**Apple Watch Ultra 2** (₹89,900)
• Most advanced: Titanium case, 36+ hour battery
• 100m water resistance, extreme durability

**Recommendation for ₹{budget:,}:**
• Series 9 if you want latest mainstream features
• Ultra 2 if you need extreme durability and longest battery

*For current exact pricing, promotions, and availability across all models, I can check Apple.com for you.*"""
    
    def _get_budget_context(self, budget: int) -> str:
        """Get contextual budget advice"""
        if budget < 25000:
            return "Consider the Apple Watch SE for best value at this price point."
        elif budget <= 50000:
            return "Apple Watch Series 9 offers the best overall experience in this range."
        else:
            return "You have access to the complete Apple Watch lineup including Ultra 2."

class AppleWatchExpert:
    """Apple Watch Expert with MANDATORY web checking when local data missing"""
    
    def __init__(self):
        self.groq_client = None
        self.selected_model = None
        self.hf_pipeline = None
        self.ollama_available = False
        
        # Initialize knowledge base with web integration
        self.knowledge_base = AppleWatchKnowledgeBase()
        
        # Status tracking
        self.active_model = "none"
        self.model_status = {}
        
        # Initialize models
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize models with proper fallback priority"""
        
        # 1. Try Groq first
        if config.is_groq_available():
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=config.get_groq_api_key())
                self.selected_model = config.test_and_select_best_model()
                
                if self.selected_model:
                    self.active_model = "groq"
                    self.model_status = {
                        "type": "groq",
                        "model": self.selected_model,
                        "status": "active",
                        "description": "Premium AI with Apple.com integration"
                    }
                    logger.info(f"✅ Groq initialized: {self.selected_model}")
                    return
                
            except Exception as e:
                logger.error(f"❌ Groq initialization failed: {e}")
        
        # 2. Try Hugging Face
        if TORCH_AVAILABLE:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                
                model_name = "microsoft/DialoGPT-small"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                self.hf_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,
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
                    "description": "Local AI with Apple.com integration"
                }
                logger.info("✅ Hugging Face initialized with Apple.com checking")
                return
                
            except Exception as e:
                logger.error(f"❌ Hugging Face initialization failed: {e}")
        
        # 3. Try Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                self.ollama_available = True
                self.active_model = "ollama"
                self.model_status = {
                    "type": "ollama",
                    "model": "llama3.1:8b",
                    "status": "active", 
                    "description": "Advanced AI with Apple.com integration"
                }
                logger.info("✅ Ollama available with Apple.com checking")
                return
                
        except Exception as e:
            logger.warning(f"❌ Ollama not available: {e}")
        
        # 4. Fallback to knowledge base with web checking
        self.active_model = "expert"
        self.model_status = {
            "type": "expert",
            "model": "web_integrated_knowledge",
            "status": "active",
            "description": "Expert knowledge with Apple.com verification"
        }
        logger.info("✅ Expert mode with mandatory Apple.com checking")
    
    def generate_apple_watch_response(self, question: str, context: str = "", 
                                sentiment: Optional[SentimentAnalysis] = None,
                                chat_history: List[Dict] = None) -> str:
        """Generate clean response - web data first, then local fallback"""
    
        try:
            # Handle non-existent products first
            if self.knowledge_base._is_non_existent_product(question.lower()):
                return self.knowledge_base._handle_non_existent_product(question)
            
            # Try web scraping first - return raw data if found
            if self.knowledge_base.web_available:
                web_data = self.knowledge_base._get_web_data(question)
                if web_data and len(web_data.strip()) > 30:
                    # Return web data directly - no formatting
                    return web_data.strip()
            
            # Fallback to local knowledge
            local_response = self.knowledge_base._get_local_response(question.lower())
            if local_response:
                return local_response.strip()
            
            # Final fallback
            return "Sorry, I couldn't find specific information about that. Please try asking about Apple Watch SE, Series 10, or Ultra 2."
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I can help with Apple Watch questions. What would you like to know?"
        
    
    def _enhance_with_groq(self, question: str, base_response: str, data: Dict, 
                          sentiment: Optional[SentimentAnalysis], 
                          chat_history: List[Dict]) -> str:
        """Enhance response using Groq while preserving web-sourced data"""
        
        # If response is from web, preserve it exactly
        if data.get("source") == "web_primary":
            return base_response
        
        # Only enhance local responses
        system_prompt = """You are an Apple Watch expert. Enhance the provided response to be more helpful and conversational while keeping all factual information exactly as provided. Do not add new product information or pricing."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nResponse to enhance: {base_response}"}
            ]
            
            response = self.groq_client.chat.completions.create(
                model=self.selected_model,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
                top_p=0.9
            )
            
            enhanced = response.choices[0].message.content.strip()
            
            if len(enhanced) > 50 and "I don't have" not in enhanced:
                return enhanced
            else:
                return base_response
                
        except Exception as e:
            logger.error(f"Groq enhancement failed: {e}")
            return base_response
    
    def _enhance_with_ollama(self, question: str, base_response: str, data: Dict,
                           sentiment: Optional[SentimentAnalysis]) -> str:
        """Enhance with Ollama while preserving web data"""
        
        # If response is from web, don't modify
        if data.get("source") == "web_primary":
            return base_response
        
        return base_response  # Keep simple for now
    
    def get_model_status(self) -> Dict[str, Union[str, dict]]:
        """Get current model status with web integration info"""
        return {
            "active_model": self.active_model,
            "status": self.model_status,
            "groq_available": bool(self.groq_client),
            "hf_available": bool(self.hf_pipeline),
            "ollama_available": self.ollama_available,
            "web_scraping_available": self.knowledge_base.web_available
        }