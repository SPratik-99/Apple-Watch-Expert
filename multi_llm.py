"""
Updated Multi-LLM System with Natural Direct Responses
Generates natural responses without "according to data" phrases
"""
import logging
from typing import List, Dict, Optional, Union
import os
import json
import requests
import time

# Only import transformers if needed (avoid bus errors)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from config import config
from classifier import SentimentAnalysis
from apple_search import RealAppleWebScraper

logger = logging.getLogger(__name__)

class AppleWatchKnowledgeBase:
    """Apple Watch knowledge with web scraping fallback"""
    
    def __init__(self):
        # Local knowledge base for instant responses
        self.local_data = {
            "models_detailed": {
                "Apple Watch SE (2nd gen)": {
                    "prices": {"40mm_gps": 24900, "44mm_gps": 28900, "40mm_cellular": 30900, "44mm_cellular": 34900},
                    "processor": "S8 SiP dual-core",
                    "health_features": ["Heart rate monitoring", "Sleep tracking", "Fall detection", "Crash detection"],
                    "key_differences": "Missing: Always-On display, ECG, Blood Oxygen, Double Tap"
                },
                "Apple Watch Series 9": {
                    "prices": {"41mm_gps": 41900, "45mm_gps": 44900, "41mm_cellular": 50900, "45mm_cellular": 53900},
                    "processor": "S9 SiP (60% faster than S8)",
                    "unique_features": ["Double Tap gesture", "On-device Siri", "Brightest display ever"],
                    "health_features": ["All SE features", "ECG app", "Blood Oxygen app", "Temperature sensing"]
                },
                "Apple Watch Ultra 2": {
                    "prices": {"49mm_cellular": 89900},
                    "processor": "S9 SiP optimized for endurance", 
                    "unique_features": ["Action Button", "Precision dual-frequency GPS", "86dB Siren", "100m water resistance"],
                    "battery": "36 hours normal, 72 hours Low Power Mode"
                }
            },
            "troubleshooting": {
                "wont_charge": ["Clean charging contacts", "Use original Apple cable", "Check power adapter", "Restart watch"],
                "connection_issues": ["Keep iPhone nearby", "Check Bluetooth", "Restart both devices", "Re-pair if necessary"]
            }
        }
        
        # Initialize real web scraper for live data
        self.web_scraper = RealAppleWebScraper()
    
    def get_context_for_query(self, query: str, use_web_fallback: bool = True) -> str:
        """Get relevant context - local first, then web scraping if needed"""
        query_lower = query.lower()
        context_parts = []
        found_local_data = False
        
        # First, try to find information in local data
        if "se" in query_lower and "apple watch se" not in query_lower:
            se_info = self.local_data["models_detailed"]["Apple Watch SE (2nd gen)"]
            context_parts.append(f"Apple Watch SE: {json.dumps(se_info, indent=2)}")
            found_local_data = True
        
        if "series 9" in query_lower or "s9" in query_lower:
            s9_info = self.local_data["models_detailed"]["Apple Watch Series 9"]
            context_parts.append(f"Apple Watch Series 9: {json.dumps(s9_info, indent=2)}")
            found_local_data = True
        
        if "ultra" in query_lower:
            ultra_info = self.local_data["models_detailed"]["Apple Watch Ultra 2"]
            context_parts.append(f"Apple Watch Ultra 2: {json.dumps(ultra_info, indent=2)}")
            found_local_data = True
        
        # Check for troubleshooting info
        if any(word in query_lower for word in ["problem", "issue", "fix", "charge"]):
            context_parts.append(f"Troubleshooting: {json.dumps(self.local_data['troubleshooting'], indent=2)}")
            found_local_data = True
        
        # If we didn't find comprehensive local data, or user asks for latest/current info, scrape web
        if (not found_local_data or 
            len(" ".join(context_parts)) < 200 or 
            any(word in query_lower for word in ["latest", "current", "new", "recent", "today", "now", "updated"])):
            
            if use_web_fallback:
                try:
                    logger.info(f"Fetching current information for '{query}'...")
                    web_context = self.web_scraper.search_apple_watch_info(query)
                    if web_context and len(web_context) > 50:
                        context_parts.append(web_context)
                        logger.info("Retrieved current information successfully")
                    else:
                        logger.warning("Web scraping returned minimal data")
                except Exception as e:
                    logger.error(f"Web scraping failed: {e}")
        
        return "\n\n".join(context_parts) if context_parts else ""

class DirectResponseAppleWatchExpert:
    """Apple Watch expert with natural, direct responses"""
    
    def __init__(self):
        self.knowledge_base = AppleWatchKnowledgeBase()
        
        # LLM clients
        self.groq_client = None
        self.selected_model = None
        self.model_info = {}
        self.hf_pipeline = None
        
        # Status tracking
        self.available_models = []
        self.web_connection_status = "unknown"
        
        self.initialize_models()
        self.test_web_connection()
    
    def test_web_connection(self):
        """Test web scraping capability"""
        try:
            if self.knowledge_base.web_scraper.test_connection():
                self.web_connection_status = "connected"
                logger.info("Apple.com connection successful")
            else:
                self.web_connection_status = "failed"
                logger.warning("Apple.com connection failed - using local data only")
        except Exception as e:
            self.web_connection_status = "error"
            logger.error(f"Web connection test error: {e}")
    
    def initialize_models(self):
        """Initialize models with automatic selection"""
        
        # 1. Initialize and test Groq models (Priority 1)
        if config.is_groq_available():
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=config.GROQ_API_KEY)
                
                # Test and select best model
                self.selected_model = config.test_and_select_best_model()
                
                if self.selected_model:
                    self.available_models.append("groq")
                    self.model_info = config.get_model_info(self.selected_model)
                    logger.info(f"Selected Groq model: {self.selected_model}")
                else:
                    logger.warning("No Groq models available")
                    
            except Exception as e:
                logger.error(f"Groq initialization failed: {e}")
        
        # 2. Initialize Hugging Face (Priority 2)
        if TORCH_AVAILABLE:
            try:
                from transformers import pipeline
                
                model_name = "microsoft/DialoGPT-small"
                
                self.hf_pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    device=-1,  # Force CPU
                    do_sample=True,
                    temperature=0.3,
                    max_new_tokens=150,
                    truncation=True,
                    pad_token_id=50256,
                    return_full_text=False
                )
                self.available_models.append("huggingface")
                logger.info(f"Hugging Face model ready: {model_name} (CPU)")
            except Exception as e:
                logger.error(f"Hugging Face init failed: {e}")
        
        # 3. Try Ollama (Priority 3)
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.available_models.append("ollama")
                logger.info("Ollama available")
        except:
            pass
        
        logger.info(f"Available models: {self.available_models}")
    
    def generate_apple_watch_response(self, question: str, context: str = "", 
                                    sentiment: Optional[SentimentAnalysis] = None,
                                    chat_history: List[Dict] = None) -> str:
        """Generate natural, direct response"""
        
        # Get context - this will use web scraping if local data is insufficient
        apple_context = self.knowledge_base.get_context_for_query(
            question, 
            use_web_fallback=(self.web_connection_status == "connected")
        )
        
        # Combine contexts intelligently
        full_context = []
        if context:
            full_context.append(f"Additional context: {context}")
        if apple_context:
            full_context.append(f"Apple Watch information: {apple_context}")
        
        combined_context = "\n\n".join(full_context)
        
        # Try models in order of preference
        for model in self.available_models:
            try:
                if model == "groq":
                    return self._generate_groq_response(question, combined_context, sentiment, chat_history)
                elif model == "huggingface":
                    return self._generate_hf_response(question, combined_context, sentiment)
                elif model == "ollama":
                    return self._generate_ollama_response(question, combined_context, sentiment)
            except Exception as e:
                logger.error(f"{model} generation failed: {e}")
                continue
        
        # Final fallback
        return self._generate_direct_response(question, sentiment, apple_context)
    
    def _generate_groq_response(self, question: str, context: str, 
                               sentiment: Optional[SentimentAnalysis], 
                               chat_history: List[Dict]) -> str:
        """Generate direct response using Groq model"""
        
        # Direct, natural system prompt
        system_prompt = """You are an Apple Watch expert. Provide direct, helpful responses without mentioning data sources.

IMPORTANT GUIDELINES:
- Give direct, natural answers as if you're an expert consultant
- Never say "according to data", "based on information provided", or similar phrases
- Speak confidently about Apple Watch models, pricing, and features
- Provide specific recommendations and comparisons
- Give step-by-step help for technical issues
- Use exact pricing and specifications when available
- Be conversational but professional"""

        # Add sentiment awareness
        if sentiment:
            if sentiment.emotion == "frustrated":
                system_prompt += "\n\nUser seems frustrated - be extra helpful and empathetic."
            elif sentiment.urgency == "high":
                system_prompt += "\n\nUser needs urgent help - provide immediate, clear solutions."
        
        # Create user message with context (hidden from response)
        if context:
            user_message = f"Context (for your knowledge only, don't mention this in response):\n{context[:1500]}\n\nUser question: {question}"
        else:
            user_message = f"User question: {question}"
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent chat history for context
        if chat_history:
            messages.extend(chat_history[-3:])
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate with selected model
        response = self.groq_client.chat.completions.create(
            model=self.selected_model,
            messages=messages,
            temperature=0.1,
            max_tokens=700,
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_hf_response(self, question: str, context: str, 
                             sentiment: Optional[SentimentAnalysis]) -> str:
        """Generate response using Hugging Face"""
        
        # Simple, direct prompt
        prompt = f"Apple Watch Expert\n\nInfo: {context[:400] if context else 'Apple Watch knowledge'}\n\nQ: {question}\nA:"
        
        try:
            result = self.hf_pipeline(
                prompt,
                max_new_tokens=120,
                temperature=0.3,
                do_sample=True,
                truncation=True
            )
            
            if result and len(result) > 0:
                response = result[0]['generated_text'].strip()
                return response if len(response) > 15 else self._generate_direct_response(question, sentiment, context)
            
        except Exception as e:
            logger.error(f"HF generation error: {e}")
        
        return self._generate_direct_response(question, sentiment, context)
    
    def _generate_ollama_response(self, question: str, context: str, 
                                 sentiment: Optional[SentimentAnalysis]) -> str:
        """Generate response using Ollama"""
        
        prompt = f"""You are an Apple Watch expert. Answer directly and naturally without mentioning data sources.

Information: {context[:1000] if context else 'Apple Watch models and features'}

Question: {question}

Direct expert answer:"""
        
        payload = {
            "model": "llama3.1:8b",
            "prompt": prompt,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 350
            },
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=25
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        
        raise Exception("Ollama request failed")
    
    def _generate_direct_response(self, question: str, 
                                 sentiment: Optional[SentimentAnalysis],
                                 context: str = "") -> str:
        """Generate direct response from knowledge base"""
        
        q_lower = question.lower()
        
        # Budget questions with direct recommendations
        if any(word in q_lower for word in ["budget", "price", "cost", "buy", "suggest", "recommend"]):
            import re
            budget_match = re.search(r'(\d+)k|(\d{4,6})', q_lower)
            if budget_match:
                budget = int(budget_match.group(1) or budget_match.group(2))
                if 'k' in q_lower and budget < 1000:
                    budget *= 1000
                
                if budget < 25000:
                    return f"""For ₹{budget:,}, the Apple Watch SE starts at ₹24,900. Here are your options:

**Stretch Budget (Recommended)**: The SE at ₹24,900 offers incredible value with heart rate monitoring, GPS, crash detection, and all essential Apple Watch features.

**Refurbished Option**: Look for certified refurbished Apple Watch SE in the ₹18,000-22,000 range from authorized resellers.

**Wait for Sales**: Festival sales can bring the SE down to ₹22,000-23,000.

The SE is really worth the extra ₹{24900-budget:,} - no other smartwatch at ₹{budget:,} comes close to the Apple Watch experience."""

                elif budget <= 35000:
                    return f"""Perfect! The Apple Watch SE is ideal for your ₹{budget:,} budget.

**Apple Watch SE (2nd gen)**:
• 40mm GPS: ₹24,900 ✅
• 44mm GPS: ₹28,900 ✅  
• 40mm Cellular: ₹30,900 ✅

**What you get**:
• Heart rate monitoring with alerts
• Sleep tracking with REM, Core, Deep stages  
• Crash detection and fall detection
• GPS for accurate workout tracking
• 85+ workout types
• 18+ hour battery life
• Water resistant to 50 meters

You'll have ₹{budget-24900:,} left over for a premium Sport Loop or Leather band. The SE gives you 90% of the full Apple Watch experience at an excellent price."""

                elif budget <= 60000:
                    return f"""Excellent! For ₹{budget:,}, I recommend the Apple Watch Series 9.

**Apple Watch Series 9**:
• 41mm GPS: ₹41,900 ✅
• 45mm GPS: ₹44,900 ✅
• 41mm Cellular: ₹50,900 ✅

**Why Series 9 is perfect for your budget**:
• **S9 SiP chip**: 60% faster than SE, smoother performance
• **Always-On Retina display**: 2000 nits, brightest Apple Watch ever
• **Double Tap**: Revolutionary gesture control - tap thumb and finger
• **Complete health suite**: ECG readings, Blood Oxygen monitoring, temperature sensing
• **Enhanced Siri**: On-device processing for privacy and speed

The ₹17,000 difference from SE gets you Always-On display, advanced health monitoring, and the latest gesture controls."""

                else:
                    return f"""With ₹{budget:,}, you have premium options:

**Option 1: Apple Watch Series 9 Complete Setup** (₹55,000-65,000)
• Series 9 Cellular 45mm: ₹53,900
• Premium Milanese Loop: ₹27,900
• Perfect for professional use with luxury feel

**Option 2: Apple Watch Ultra 2** (₹89,900)  
• Most advanced Apple Watch ever
• 49mm titanium case, extreme durability
• 36+ hour battery, 100m water resistance
• Perfect for athletes and outdoor adventures

**My recommendation**: Series 9 unless you specifically need Ultra's extreme features like diving capability or multi-day battery life."""
        
        # Model comparisons
        elif any(word in q_lower for word in ["compare", "vs", "versus", "difference"]):
            if "se" in q_lower and ("series 9" in q_lower or "s9" in q_lower):
                return """**Apple Watch SE vs Series 9 - Complete Comparison**

**Apple Watch SE (₹24,900)**
✅ Heart rate monitoring, sleep tracking
✅ Crash detection, fall detection  
✅ GPS, fitness tracking, 85+ workouts
✅ 18+ hour battery life
❌ No Always-On display
❌ No ECG or Blood Oxygen
❌ No Double Tap gesture
❌ No temperature sensing

**Apple Watch Series 9 (₹41,900)**
✅ All SE features PLUS:
✅ Always-On Retina display (2000 nits)
✅ ECG app for heart rhythm monitoring
✅ Blood Oxygen app for wellness insights
✅ Double Tap gesture control
✅ S9 chip (60% faster than SE)
✅ Temperature sensing for health tracking
✅ On-device Siri processing

**₹17,000 Difference - Worth It For:**
• Health enthusiasts who want ECG and Blood Oxygen monitoring
• Users who want Always-On display convenience
• Power users who need faster performance
• People who love having the latest technology

**Stick with SE if:** You want essential Apple Watch features, are budget-conscious, or this is your first Apple Watch."""

            elif "ultra" in q_lower and ("series 9" in q_lower or "s9" in q_lower):
                return """**Apple Watch Series 9 vs Ultra 2 - Which One?**

**Apple Watch Series 9 (₹41,900-53,900)**
• **Best for**: Most users wanting complete Apple Watch experience
• **Display**: Always-On Retina, 2000 nits brightness
• **Health**: ECG, Blood Oxygen, temperature sensing, all standard features
• **Battery**: 18 hours normal, 36 hours Low Power Mode
• **Build**: Aluminum/steel, 50m water resistance

**Apple Watch Ultra 2 (₹89,900)**
• **Best for**: Extreme sports, outdoor adventures, professionals in harsh conditions
• **Display**: Largest (49mm), 3000 nits brightness, flat sapphire crystal
• **Build**: Aerospace titanium, 100m water resistance, military-grade durability
• **Battery**: 36 hours normal, 72 hours Low Power Mode
• **Special**: Action Button, precision dual-frequency GPS, 86dB emergency siren

**Ultra 2 worth ₹36,000+ extra only if you:**
• Do serious multi-day outdoor adventures
• Need maximum durability for extreme conditions
• Dive regularly (up to 40m capability)
• Need longest possible battery life
• Work in harsh environments

**For 95% of users, Series 9 is the perfect choice.**"""
        
        # Troubleshooting with direct solutions
        elif any(word in q_lower for word in ["problem", "fix", "not working", "charge", "broken"]):
            if "charge" in q_lower:
                return """**Apple Watch Charging Fix - Step by Step**

**Step 1: Clean Everything**
• Remove watch from charger completely
• Use soft, lint-free cloth (microfiber works best)
• Clean the circular area on watch back thoroughly
• Clean charger surface - remove any debris or moisture

**Step 2: Check Your Setup**
• Use ONLY original Apple charging cable (or certified MFi)
• Connect to 5W+ USB power adapter (iPhone charger works)
• Make sure cable clicks magnetically to watch back
• You should see green lightning bolt appear on watch

**Step 3: Force Restart Your Watch**
• Hold side button + Digital Crown together
• Keep holding for exactly 10 seconds
• Release when Apple logo appears
• Place back on charger after restart

**Step 4: Try Different Positions**
• Test different orientations on charger
• Ensure watch lies completely flat
• Check for proper magnetic alignment
• Try different power outlet

**Still not working?** Try a different USB adapter, check cable for damage, or contact Apple Support if under warranty.

**Prevention tip**: Clean charging contacts weekly and avoid charging in extreme temperatures."""
        
        # Default expert response
        return """I'm your Apple Watch expert! I can help with:

**🏷️ Model Recommendations**
Find the perfect Apple Watch for your budget and needs

**⚖️ Detailed Comparisons**  
SE vs Series 9 vs Ultra 2 with exact differences

**🔧 Technical Support**
Step-by-step solutions for setup, pairing, and troubleshooting

**💰 Current Pricing**
Official Apple pricing and best value recommendations

**❤️ Health Features**
ECG, heart rate, blood oxygen, sleep tracking guidance

What specific Apple Watch question can I help you with?"""
    
    def get_model_status(self) -> Dict[str, Union[str, dict]]:
        """Get model status"""
        status = {}
        
        # AI model status
        if "groq" in self.available_models:
            status["groq"] = "available"
            status["selected_model"] = self.selected_model
            status["model_info"] = self.model_info
        else:
            status["groq"] = "unavailable"
            status["selected_model"] = None
        
        for model in ["huggingface", "ollama"]:
            status[model] = "available" if model in self.available_models else "unavailable"
        
        # Web scraping status
        status["web_scraping"] = self.web_connection_status
        
        return status

# Global direct response expert instance
direct_response_expert = DirectResponseAppleWatchExpert()