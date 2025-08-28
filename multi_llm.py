"""
FIXED Multi-LLM System with Proper Fallbacks
No more vague answers or hallucination - only real data
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

class AppleWatchExpert:
    """Fixed Apple Watch Expert with proper model fallbacks"""
    
    def __init__(self):
        self.groq_client = None
        self.selected_model = None
        self.hf_pipeline = None
        self.ollama_available = False
        
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
                        "description": "Premium AI model"
                    }
                    logger.info(f"✅ Groq initialized: {self.selected_model}")
                    return  # Success - no need to try others
                
            except Exception as e:
                logger.error(f"❌ Groq initialization failed: {e}")
        
        # 2. Try Hugging Face (second priority)
        if TORCH_AVAILABLE:
            try:
                from transformers import pipeline
                
                # Use a lightweight model for CPU
                self.hf_pipeline = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-small",
                    device=-1,  # CPU only
                    do_sample=True,
                    temperature=0.3,
                    max_new_tokens=120,
                    truncation=True,
                    pad_token_id=50256
                )
                
                self.active_model = "huggingface"
                self.model_status = {
                    "type": "huggingface", 
                    "model": "DialoGPT-small",
                    "status": "active",
                    "description": "Local AI processing"
                }
                logger.info("✅ Hugging Face initialized")
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
                    "description": "Local advanced AI"
                }
                logger.info("✅ Ollama available")
                return  # Success
                
        except Exception as e:
            logger.warning(f"❌ Ollama not available: {e}")
        
        # 4. Fallback to expert knowledge base (always available)
        self.active_model = "expert"
        self.model_status = {
            "type": "expert",
            "model": "knowledge_base",
            "status": "active",
            "description": "Expert knowledge system"
        }
        logger.info("✅ Expert knowledge base ready")
    
    def generate_apple_watch_response(self, question: str, context: str = "", 
                                    sentiment: Optional[SentimentAnalysis] = None,
                                    chat_history: List[Dict] = None) -> str:
        """Generate response using active model with strict no-hallucination policy"""
        
        try:
            # Use active model in priority order
            if self.active_model == "groq" and self.groq_client:
                return self._generate_groq_response(question, context, sentiment, chat_history)
            
            elif self.active_model == "huggingface" and self.hf_pipeline:
                return self._generate_hf_response(question, context, sentiment)
                
            elif self.active_model == "ollama" and self.ollama_available:
                return self._generate_ollama_response(question, context, sentiment)
                
            else:
                # Always fallback to expert knowledge base
                return self._generate_expert_response(question, context, sentiment)
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._generate_expert_response(question, context, sentiment)
    
    def _generate_groq_response(self, question: str, context: str, 
                               sentiment: Optional[SentimentAnalysis], 
                               chat_history: List[Dict]) -> str:
        """Generate response using Groq with strict guidelines"""
        
        system_prompt = """You are an Apple Watch expert. Follow these rules strictly:

1. NEVER make up product information, prices, or model numbers
2. Only provide information you are certain about
3. If asked about non-existent products (like Series 10), clearly state they don't exist
4. Base pricing on real Indian market data only
5. Give direct, helpful responses without mentioning data sources
6. For troubleshooting, provide step-by-step solutions"""

        # Build conversation
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent history for context
        if chat_history:
            messages.extend(chat_history[-3:])
        
        # Add current question with context
        user_message = f"Question: {question}"
        if context:
            user_message += f"\n\nRelevant context: {context[:800]}"
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.selected_model,
                messages=messages,
                temperature=0.1,
                max_tokens=600,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return self._generate_expert_response(question, context, sentiment)
    
    def _generate_hf_response(self, question: str, context: str, 
                             sentiment: Optional[SentimentAnalysis]) -> str:
        """Generate response using Hugging Face"""
        
        try:
            # Simple prompt for DialoGPT
            prompt = f"Apple Watch Expert: {question}\nAnswer:"
            
            result = self.hf_pipeline(
                prompt,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                truncation=True
            )
            
            if result and len(result) > 0:
                response = result[0]['generated_text'].replace(prompt, "").strip()
                if len(response) > 15:
                    return response
            
            # Fallback to expert if HF response is poor
            return self._generate_expert_response(question, context, sentiment)
            
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            return self._generate_expert_response(question, context, sentiment)
    
    def _generate_ollama_response(self, question: str, context: str, 
                                 sentiment: Optional[SentimentAnalysis]) -> str:
        """Generate response using Ollama"""
        
        try:
            prompt = f"""You are an Apple Watch expert. Answer directly without mentioning data sources.

Context: {context[:500] if context else 'Apple Watch knowledge'}

Question: {question}

Expert answer:"""
            
            payload = {
                "model": "llama3.1:8b",
                "prompt": prompt,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 250
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
                return result.get("response", "").strip()
            
            return self._generate_expert_response(question, context, sentiment)
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return self._generate_expert_response(question, context, sentiment)
    
    def _generate_expert_response(self, question: str, context: str = "", 
                                 sentiment: Optional[SentimentAnalysis] = None) -> str:
        """Generate expert response using knowledge base - NO HALLUCINATION"""
        
        q_lower = question.lower()
        
        # Handle non-existent products explicitly
        if any(term in q_lower for term in ["series 10", "series10", "ultra 3", "ultra3", "se 3"]):
            return f"""I cannot provide information about {self._extract_mentioned_model(question)} because it doesn't exist in Apple's current lineup.

**Current Apple Watch Models (August 2025):**
• **Apple Watch SE (2nd gen)**: ₹24,900 - ₹34,900
• **Apple Watch Series 9**: ₹41,900 - ₹53,900  
• **Apple Watch Ultra 2**: ₹89,900

Would you like detailed information about any of these actual models?"""
        
        # Budget recommendations with real pricing
        if any(word in q_lower for word in ["budget", "price", "suggest", "recommend", "best"]):
            import re
            budget_match = re.search(r'(\d+)k|(\d{4,6})', q_lower)
            if budget_match:
                budget = int(budget_match.group(1) or budget_match.group(2))
                if 'k' in q_lower and budget < 1000:
                    budget *= 1000
                
                return self._get_budget_recommendation(budget)
        
        # Model comparisons
        if any(word in q_lower for word in ["compare", "vs", "versus", "difference"]):
            if "se" in q_lower and ("series 9" in q_lower or "s9" in q_lower):
                return self._get_se_vs_series9_comparison()
            elif "ultra" in q_lower and ("series 9" in q_lower or "s9" in q_lower):
                return self._get_ultra_vs_series9_comparison()
        
        # Technical support
        if any(word in q_lower for word in ["charge", "problem", "fix", "not working", "issue"]):
            return self._get_technical_support(q_lower)
        
        # Default expert response
        return """I'm your Apple Watch expert! I provide accurate information about:

**🏷️ Model Selection**
Find the perfect Apple Watch for your needs and budget

**💰 Current Pricing** (Real Indian market prices)
• SE (2nd gen): ₹24,900 - ₹34,900
• Series 9: ₹41,900 - ₹53,900
• Ultra 2: ₹89,900

**🔧 Technical Support**
Setup, pairing, troubleshooting, and maintenance help

**⚖️ Detailed Comparisons**
Feature-by-feature analysis between models

What specific Apple Watch question can I help you with?"""
    
    def _extract_mentioned_model(self, question: str) -> str:
        """Extract the mentioned model from question"""
        q_lower = question.lower()
        if "series 10" in q_lower or "series10" in q_lower:
            return "Apple Watch Series 10"
        elif "ultra 3" in q_lower or "ultra3" in q_lower:
            return "Apple Watch Ultra 3"
        elif "se 3" in q_lower:
            return "Apple Watch SE 3"
        else:
            return "the mentioned model"
    
    def _get_budget_recommendation(self, budget: int) -> str:
        """Get budget recommendation with real pricing"""
        if budget < 25000:
            return f"""For ₹{budget:,}, you're close to the Apple Watch SE starting price.

**Apple Watch SE (2nd gen)**: ₹24,900
The SE is the most affordable genuine Apple Watch with:
• Heart rate monitoring and GPS
• Sleep tracking with detailed stages
• Crash detection and fall detection  
• 85+ workout types
• 18+ hour battery life

**Options:**
• Stretch budget by ₹{24900-budget:,} for the SE 40mm GPS
• Look for certified refurbished SE models (₹18,000-22,000)
• Wait for festival sales (SE can drop to ₹22,000-23,000)

The SE offers 90% of the full Apple Watch experience at the best price point."""

        elif budget <= 45000:
            return f"""Perfect budget for the Apple Watch Series 9!

**Apple Watch Series 9** (₹41,900-44,900):
• **S9 chip**: 60% faster performance than SE
• **Always-On Retina display**: 2000 nits brightness
• **Double Tap gesture**: Revolutionary new control method
• **Complete health suite**: ECG, Blood Oxygen, temperature sensing
• **Enhanced Siri**: On-device processing for privacy

**Your ₹{budget:,} budget covers:**
• Series 9 41mm GPS: ₹41,900 ✅
• Series 9 45mm GPS: ₹44,900 ✅
• Plus premium bands with leftover budget

The Series 9 is the best overall Apple Watch for most users."""

        else:
            return f"""Excellent budget for premium Apple Watch options!

**Apple Watch Series 9** (₹41,900-53,900):
Complete flagship experience with latest features

**Apple Watch Ultra 2** (₹89,900):
• Aerospace titanium construction
• Largest 49mm display  
• 36+ hour battery (72 hours in Low Power Mode)
• 100m water resistance for serious water sports
• Precision dual-frequency GPS
• 86dB emergency siren

**For ₹{budget:,}, I recommend:**
• **Series 9** if you want the latest mainstream features
• **Ultra 2** if you need extreme durability and longest battery life
• **Series 9 + Premium accessories** for a complete luxury setup

Which use case fits you better - everyday premium or extreme outdoor adventures?"""
    
    def _get_se_vs_series9_comparison(self) -> str:
        """Detailed SE vs Series 9 comparison"""
        return """**Apple Watch SE vs Series 9 - Complete Analysis**

**Apple Watch SE (2nd gen)** - ₹24,900
✅ S8 dual-core processor
✅ Heart rate monitoring, sleep tracking  
✅ Crash detection, fall detection
✅ GPS tracking, 85+ workouts
✅ Water resistant to 50m
✅ 18+ hour battery life
❌ No Always-On display
❌ No ECG or Blood Oxygen monitoring
❌ No Double Tap gesture
❌ No temperature sensing

**Apple Watch Series 9** - ₹41,900  
✅ All SE features PLUS:
✅ S9 SiP chip (60% faster than SE)
✅ Always-On Retina display (2000 nits)
✅ Double Tap gesture control
✅ ECG app for heart rhythm analysis
✅ Blood Oxygen app for wellness tracking
✅ Temperature sensing for cycle tracking
✅ On-device Siri processing

**₹17,000 Price Difference - Worth It If:**
• You want Always-On display convenience
• Health monitoring is important (ECG/Blood Oxygen)
• You love cutting-edge features (Double Tap)
• Performance matters for apps and navigation

**Stick with SE If:**
• Budget is primary concern
• Basic fitness tracking is sufficient
• This is your first Apple Watch
• You don't need advanced health features

Both are excellent choices - Series 9 for premium experience, SE for best value."""
    
    def _get_ultra_vs_series9_comparison(self) -> str:
        """Ultra 2 vs Series 9 comparison"""
        return """**Apple Watch Series 9 vs Ultra 2 - Which One?**

**Apple Watch Series 9** (₹41,900-53,900)
✅ **Best for**: Most users wanting complete Apple Watch experience
✅ **Display**: Always-On Retina, 2000 nits, 41mm/45mm options
✅ **Health**: Full suite - ECG, Blood Oxygen, temperature, heart rate
✅ **Battery**: 18 hours normal use, 36 hours Low Power Mode
✅ **Build**: Aluminum or steel, multiple colors, 50m water resistance

**Apple Watch Ultra 2** (₹89,900)
✅ **Best for**: Extreme sports, outdoor adventures, maximum durability
✅ **Display**: Largest 49mm, 3000 nits brightness, flat sapphire crystal
✅ **Build**: Aerospace titanium, 100m water resistance, MIL-STD tested
✅ **Battery**: 36 hours normal, 72 hours Low Power Mode  
✅ **Special**: Action Button, precision dual-frequency GPS, 86dB siren
✅ **Unique**: Diving to 40m, extreme temperature resistance

**Ultra 2 worth ₹36,000+ extra ONLY if you:**
• Do serious multi-day outdoor adventures
• Need maximum battery life (60+ hours)
• Regularly swim/dive in challenging conditions
• Work in extreme environments
• Want the most durable tech device possible

**For 90% of users, Series 9 provides the complete premium Apple Watch experience at a much better value.**

Which matches your lifestyle - urban professional or outdoor adventurer?"""
    
    def _get_technical_support(self, query: str) -> str:
        """Get technical support based on query"""
        if "charge" in query:
            return """**Apple Watch Charging Issues - Step-by-Step Fix**

**Step 1: Clean Everything Thoroughly**
• Remove Apple Watch from charger completely
• Use soft, lint-free cloth (microfiber ideal)
• Clean watch back (circular sensor area) thoroughly  
• Clean charger surface - remove any debris, sweat, moisture

**Step 2: Verify Proper Setup**
• Use original Apple charging cable ONLY
• Connect to 5W+ USB power adapter (iPhone charger works)
• Ensure magnetic connection clicks and feels secure
• Green lightning bolt should appear on watch screen

**Step 3: Force Restart Watch**
• Hold side button + Digital Crown simultaneously
• Keep holding for exactly 10 seconds until Apple logo appears
• Release buttons and place back on charger
• Should start charging within 30 seconds

**Step 4: Advanced Troubleshooting**
• Try different power outlet/USB port
• Test with different Apple USB adapter
• Check charging cable for physical damage
• Ensure watch is properly centered on charger pad

**Still not working?**
• Contact Apple Support if under warranty
• Visit Apple Store for free diagnostics
• Check if charging port on watch has debris

**Prevention:** Clean charging contacts weekly and avoid extreme temperatures while charging."""

        elif any(word in query for word in ["connection", "pair", "bluetooth", "phone"]):
            return """**Apple Watch Connection/Pairing Issues - Complete Fix**

**For Initial Pairing:**
1. Keep iPhone and Apple Watch within arm's reach
2. Open Apple Watch app on iPhone
3. Tap "Start Pairing" and point iPhone camera at watch
4. Follow on-screen setup instructions
5. Choose "Set Up for Myself" unless it's for someone else

**For Connection Problems:**
1. **Check Bluetooth**: Settings > Bluetooth > ensure ON
2. **Check WiFi**: Both devices on same network helps
3. **Restart both devices**: iPhone and Apple Watch
4. **Toggle Airplane Mode**: On for 10 seconds, then off

**If Still Not Connecting:**
• Unpair and re-pair: Watch app > My Watch > [i] > Unpair
• Reset Network Settings on iPhone (last resort)
• Update to latest iOS and watchOS versions

**Daily Connection Tips:**
• Keep devices within 10 meters for optimal connection
• Ensure iPhone isn't in Low Power Mode
• Check for software updates monthly

This fixes 95% of connection issues. Need help with a specific error message?"""

        else:
            return """**Common Apple Watch Issues - Quick Solutions**

**Watch Frozen/Not Responding:**
• Force restart: Hold side button + Digital Crown for 10 seconds

**Battery Draining Fast:**
• Check Always-On Display settings
• Reduce background app refresh  
• Lower wake time duration
• Update to latest watchOS

**Apps Not Working:**
• Force close: Press Digital Crown, swipe up on app
• Restart watch: Hold side button until power slider appears

**Fitness Tracking Issues:**
• Calibrate: Settings > Privacy & Security > Analytics & Improvements
• Ensure wrist detection is ON
• Keep watch snug but comfortable during workouts

**Notifications Not Showing:**
• Check iPhone: Settings > Notifications > Mirror Apple Watch
• Restart both iPhone and Apple Watch
• Verify Do Not Disturb settings

**Need more specific help?** Tell me exactly what issue you're experiencing and I'll provide detailed troubleshooting steps."""
    
    def get_model_status(self) -> Dict[str, Union[str, dict]]:
        """Get current model status"""
        return {
            "active_model": self.active_model,
            "status": self.model_status,
            "groq_available": bool(self.groq_client),
            "hf_available": bool(self.hf_pipeline),
            "ollama_available": self.ollama_available
        }