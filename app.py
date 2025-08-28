"""
Production-Ready Apple Watch Expert
Handles deployment scenarios: with/without data files and API keys
"""
import streamlit as st
from datetime import datetime
from typing import List, Dict, Optional
import logging
import re
import time
import os
from pathlib import Path

# Import components - use updated config
from config import config
from document_loader import AppleWatchDocumentLoader
from vector_store import VectorStoreManager
from classifier import AppleWatchClassifier
from multi_llm import DirectResponseAppleWatchExpert

# Minimal logging for production
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

class ProductionAppleWatchBot:
    """Production-ready Apple Watch bot with environment detection"""
    
    def __init__(self):
        self._document_loader = None
        self._vector_store = None
        self._classifier = None
        self._expert = None
        self._components_status = {}
        self._initialization_complete = False
        
        # Detect deployment environment
        self.environment = config.get_environment()
        self.has_api_key = config.is_groq_available()
        self.data_status = config.validate_data_structure()
        
    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = AppleWatchClassifier()
        return self._classifier
    
    @property
    def expert(self):
        if self._expert is None:
            try:
                # Progress bar for initialization
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if self.has_api_key:
                    status_text.text("🚀 Initializing premium AI models...")
                    progress_bar.progress(20)
                else:
                    status_text.text("🧠 Initializing expert system...")
                    progress_bar.progress(20)
                
                time.sleep(0.3)
                
                status_text.text("🌐 Connecting to Apple.com...")
                progress_bar.progress(50)
                
                self._expert = DirectResponseAppleWatchExpert()
                
                status_text.text("✅ Ready to help!")
                progress_bar.progress(100)
                time.sleep(0.3)
                
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
                
                self._initialization_complete = True
                    
            except Exception as e:
                logger.error(f"Expert init failed: {e}")
                # Clean up progress indicators
                try:
                    progress_bar.empty()
                    status_text.empty()
                except:
                    pass
                self._expert = self._create_production_fallback()
        return self._expert
    
    def _create_production_fallback(self):
        """Production-ready fallback expert"""
        class ProductionExpert:
            def generate_apple_watch_response(self, question, context="", sentiment=None, chat_history=None):
                return self._production_response(question, context)
            
            def get_model_status(self):
                return {"status": "ready", "mode": "expert_fallback"}
            
            def _production_response(self, question, context=""):
                q_lower = question.lower()
                
                # Budget responses with production quality
                if any(word in q_lower for word in ["budget", "price", "suggest", "recommend"]):
                    import re
                    budget_match = re.search(r'(\d+)k|(\d{4,6})', q_lower)
                    if budget_match:
                        budget = int(budget_match.group(1) or budget_match.group(2))
                        if 'k' in q_lower and budget < 1000:
                            budget *= 1000
                        
                        if budget < 25000:
                            return f"""For a ₹{budget:,} budget, the Apple Watch SE starts at ₹24,900.

**Your Options:**
• **Apple Watch SE 40mm GPS**: ₹24,900 (only ₹{24900-budget:,} more)
• **Certified Refurbished SE**: Look for ₹18,000-22,000 range
• **Festival Sales**: SE often drops to ₹22,000-23,000

**Why SE is worth it**: The SE includes heart rate monitoring, GPS tracking, crash detection, sleep tracking, and all essential Apple Watch features. No other smartwatch at ₹{budget:,} comes close to the Apple Watch ecosystem."""

                        elif budget <= 60000:
                            return f"""Perfect! For ₹{budget:,}, I recommend the Apple Watch Series 9.

**Apple Watch Series 9** (₹41,900-44,900):
• S9 SiP chip - 60% faster than SE
• Always-On Retina display (2000 nits) 
• Double Tap gesture control
• Complete health suite: ECG, Blood Oxygen, temperature sensing
• Enhanced Siri with on-device processing

**Value Analysis**: The ₹17,000 difference from SE gets you Always-On display, advanced health monitoring, latest gesture controls, and significantly faster performance. Perfect for your budget with room for premium bands."""

                        else:
                            return f"""Excellent budget for premium Apple Watch options:

**Apple Watch Series 9 Complete Setup** (₹55,000-65,000):
• Series 9 Cellular 45mm: ₹53,900
• Premium Milanese Loop: ₹27,900
• Total within your ₹{budget:,} budget

**Apple Watch Ultra 2** (₹89,900):
• Most advanced Apple Watch
• 49mm titanium, extreme durability
• 36+ hour battery, precision GPS
• Perfect for outdoor adventures

**Recommendation**: Series 9 unless you specifically need Ultra's extreme features like diving capability or multi-day battery."""
                
                # Handle comparisons
                elif any(word in q_lower for word in ["compare", "vs", "versus", "difference"]):
                    if "se" in q_lower and ("series 9" in q_lower or "s9" in q_lower):
                        return """**Apple Watch SE vs Series 9 - Complete Analysis**

**Apple Watch SE (₹24,900)**
✅ Heart rate monitoring, sleep tracking
✅ Crash detection, fall detection
✅ GPS, 85+ workouts, water resistant
✅ 18+ hour battery, Emergency SOS
❌ No Always-On display
❌ No ECG or Blood Oxygen
❌ No Double Tap gesture

**Apple Watch Series 9 (₹41,900)**  
✅ All SE features PLUS:
✅ Always-On Retina display (2000 nits)
✅ ECG app for heart monitoring
✅ Blood Oxygen measurements
✅ Double Tap gesture control
✅ S9 chip (60% faster)
✅ Temperature sensing

**₹17,000 Difference Worth It If:**
• You want Always-On display convenience
• Health monitoring is important (ECG/Blood Oxygen)
• You love latest technology features
• Performance matters for apps and interactions

**Stick with SE If:**
• Budget-conscious decision
• First Apple Watch experience
• Basic fitness and notifications sufficient"""
                
                # Technical support
                elif any(word in q_lower for word in ["charge", "problem", "fix", "not working"]):
                    if "charge" in q_lower:
                        return """**Apple Watch Charging Issues - Complete Solution**

**Step 1: Clean Everything**
• Remove Apple Watch from charger
• Use soft, lint-free cloth (microfiber ideal)
• Clean watch back (circular sensor area) thoroughly
• Clean charger surface - remove debris, moisture, sweat

**Step 2: Verify Setup**
• Use original Apple charging cable only
• Connect to 5W+ power adapter (iPhone charger works)
• Ensure magnetic connection clicks properly
• Green lightning bolt should appear on watch

**Step 3: Restart Your Watch**
• Hold side button + Digital Crown simultaneously
• Keep holding for exactly 10 seconds
• Release when Apple logo appears
• Place back on charger after restart

**Step 4: Troubleshoot Further**
• Try different power outlet
• Test with different USB adapter
• Check charging cable for damage
• Ensure watch is centered on charger

**Still not working?** Contact Apple Support if under warranty, or visit Apple Store for diagnostics."""
                
                # Default professional response
                return """I'm your Apple Watch expert! I can provide comprehensive help with:

**🏷️ Smart Recommendations**
Find the perfect Apple Watch model for your specific budget and needs

**⚖️ Detailed Comparisons**  
In-depth analysis of SE vs Series 9 vs Ultra 2 with exact differences

**🔧 Technical Support**
Step-by-step troubleshooting for setup, charging, and connection issues

**💰 Current Pricing**
Official Apple pricing with value analysis and buying recommendations

**❤️ Health Features**
Complete guide to heart monitoring, ECG, sleep tracking, and wellness features

What specific Apple Watch question can I help you with today?"""
        
        return ProductionExpert()
    
    def _get_vector_store(self):
        """Get vector store with production error handling"""
        if self._vector_store is None:
            try:
                # Only show progress if we have actual documents
                if self.data_status.get("pdfs", {}).get("files", 0) > 0:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📄 Processing your documents...")
                    progress_bar.progress(25)
                    
                    self._vector_store = VectorStoreManager()
                    
                    status_text.text("🔍 Building search index...")
                    progress_bar.progress(75)
                    
                    success = self._vector_store.initialize_vector_store()
                    
                    status_text.text("✅ Documents ready!")
                    progress_bar.progress(100)
                    time.sleep(0.3)
                    
                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    self._components_status["vector_store"] = "ready" if success else "failed"
                else:
                    # No documents available - silent initialization
                    self._vector_store = VectorStoreManager()
                    success = self._vector_store.initialize_vector_store()
                    self._components_status["vector_store"] = "no_documents"
                    
            except Exception as e:
                logger.error(f"Vector store failed: {e}")
                self._components_status["vector_store"] = "failed"
        return self._vector_store
    
    def extract_budget(self, text: str) -> Optional[int]:
        """Extract budget from user input"""
        patterns = [
            r'₹\s*(\d+)k',
            r'₹\s*(\d+),?(\d+)', 
            r'(\d+)k',
            r'(\d{4,6})',
        ]
        
        text_lower = text.lower().replace(',', '').replace(' ', '')
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    if isinstance(matches[0], tuple):
                        if len(matches[0]) == 2 and matches[0][1]:
                            return int(matches[0][0] + matches[0][1])
                        else:
                            num = int(matches[0][0])
                    else:
                        num = int(matches[0])
                    
                    if 'k' in text_lower and num < 1000:
                        num *= 1000
                    
                    if 10000 <= num <= 150000:
                        return num
                except ValueError:
                    continue
        
        return None
    
    def is_budget_question(self, text: str, chat_history: List[Dict]) -> bool:
        """Check if this is a budget-related question"""
        budget_keywords = ['suggest', 'recommend', 'best', 'good', 'want', 'need', 'buy', 'purchase']
        has_budget = self.extract_budget(text) is not None
        has_keywords = any(word in text.lower() for word in budget_keywords)
        
        # Check previous context
        previous_budget_context = False
        if chat_history:
            for msg in reversed(chat_history[-3:]):
                if msg.get('role') == 'user':
                    last_msg = msg.get('content', '').lower()
                    if (self.extract_budget(msg.get('content', '')) or 
                        any(word in last_msg for word in ['budget', 'suggest', 'recommend', 'best', 'price'])):
                        previous_budget_context = True
                        break
        
        return (has_budget or 
                (has_keywords and previous_budget_context) or 
                (has_keywords and any(word in text.lower() for word in ['watch', 'apple'])))
    
    def get_response(self, user_input: str, chat_history: List[Dict] = None) -> str:
        """Get production-ready response"""
        
        try:
            # Step 1: Classify and analyze
            classification = self.classifier.classify_input(user_input)
            sentiment = self.classifier.analyze_sentiment(user_input)
            
            # Handle non-Apple Watch questions
            if not classification["is_apple_watch"] and not self.is_budget_question(user_input, chat_history):
                return self._handle_general_chat(user_input, sentiment)
            
            # Step 2: Get document context only if we have documents and complex query
            doc_context = ""
            if (self.data_status.get("pdfs", {}).get("files", 0) > 0 and
                any(word in user_input.lower() for word in ["technical", "manual", "detailed", "specification", "compare"])):
                try:
                    vector_store = self._get_vector_store()
                    if vector_store and hasattr(vector_store, 'is_ready') and vector_store.is_ready():
                        doc_results = vector_store.similarity_search(user_input, k=2)
                        if doc_results:
                            doc_context = " ".join([doc.page_content[:200] for doc in doc_results])
                except Exception as e:
                    logger.error(f"Document search failed: {e}")
            
            # Step 3: Generate response
            response = self.expert.generate_apple_watch_response(
                question=user_input,
                context=doc_context,
                sentiment=sentiment,
                chat_history=chat_history
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I can help you with any Apple Watch questions - model recommendations, comparisons, pricing, or troubleshooting. What would you like to know?"
    
    def _handle_general_chat(self, user_input: str, sentiment) -> str:
        """Handle non-Apple Watch conversation"""
        
        if sentiment.intent == "greeting":
            greeting_msg = "Hello! 👋 I'm your Apple Watch expert assistant."
            
            if self.has_api_key:
                greeting_msg += "\n\n🚀 **Enhanced with premium AI** for the most accurate responses."
            
            greeting_msg += """

I can help you with:
• **Smart recommendations** based on your budget and needs
• **Model comparisons** between all Apple Watch options  
• **Technical support** for any issues or setup questions
• **Current pricing** and buying advice

What Apple Watch question can I help you with?"""
            return greeting_msg
        
        elif sentiment.intent == "gratitude":
            return """You're very welcome! 😊

Happy to help with any other Apple Watch questions you might have!"""
        
        else:
            return """I'm your Apple Watch expert! 

I can help with:
• **Model selection** - find the perfect Apple Watch for your needs
• **Comparisons** - detailed analysis between SE, Series 9, and Ultra 2
• **Technical support** - troubleshooting and setup assistance  
• **Buying guidance** - pricing and recommendations

What Apple Watch topic can I help you with today?"""
    
    def get_system_status(self) -> Dict:
        """Get production system status"""
        try:
            status = {
                "system": "ready" if self._initialization_complete else "initializing",
                "environment": self.environment,
                "api_available": self.has_api_key,
                "documents": self._components_status.get("vector_store", "available")
            }
            
            # Add data status
            status.update(self.data_status)
            
            return status
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"system": "ready", "environment": "unknown"}

# Initialize production bot
@st.cache_resource
def get_production_bot():
    return ProductionAppleWatchBot()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

bot = get_production_bot()

def add_message(role: str, content: str):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M")
    })

def handle_user_input(user_input: str):
    if not user_input.strip():
        return
    
    add_message("user", user_input)
    
    # Get relevant chat history
    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] in ["user", "assistant"]:
            chat_history.append({
                "role": msg["role"], 
                "content": msg["content"]
            })
    
    # Generate response
    try:
        response = bot.get_response(user_input, chat_history[-6:])
        add_message("assistant", response)
    except Exception as e:
        logger.error(f"Handle input failed: {e}")
        add_message("assistant", "I can help you with any Apple Watch questions. What would you like to know?")
    
    st.rerun()

# Production Sidebar
with st.sidebar:
    st.title("⌚ Apple Watch Expert")
    
    # Environment-aware caption
    if bot.has_api_key:
        st.caption("🚀 Premium AI • Enhanced responses")
    else:
        st.caption("🧠 Expert AI • Professional answers")
    
    # Simple status
    try:
        status = bot.get_system_status()
        
        if status.get("system") == "ready":
            if bot.has_api_key:
                st.success("🟢 Premium mode active")
            else:
                st.success("🟢 Ready to help")
        else:
            st.info("🔄 Getting ready...")
        
        # Data status (only show if relevant)
        doc_status = status.get("documents", "available")
        if doc_status == "ready":
            st.caption("📄 Your documents loaded")
        elif doc_status == "no_documents":
            st.caption("📄 Using web data & knowledge base")
        else:
            st.caption("📄 Smart responses available")
            
    except:
        st.success("🟢 Ready to help")
    
    st.markdown("---")
    
    # Help section
    with st.expander("💡 What I can help with"):
        st.markdown("""
        **🏷️ Recommendations**
        • Perfect watch for any budget
        • Personalized model selection
        
        **💰 Pricing & Value**
        • Current Apple Watch prices
        • Best value analysis
        
        **🔧 Technical Support**
        • Setup and pairing help
        • Troubleshooting solutions
        
        **❤️ Health & Features**
        • Health monitoring guide
        • Feature comparisons
        """)
    
    # Environment info (for debugging)
    if bot.environment != "local":
        with st.expander("ℹ️ System Info"):
            st.caption(f"Environment: {bot.environment}")
            if bot.has_api_key:
                st.caption("AI: Premium models available")
            else:
                st.caption("AI: Expert knowledge base")
    
    # Clear chat
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Interface  
st.title("Apple Watch Expert Assistant")

# Environment-aware subtitle
if bot.has_api_key:
    st.caption("Ask me anything about Apple Watch - powered by premium AI models!")
else:
    st.caption("Ask me anything about Apple Watch - expert knowledge at your service!")

# Welcome message
if not st.session_state.messages:
    welcome_msg = "👋 **Welcome!** I'm your Apple Watch expert assistant.\n\n"
    
    if bot.has_api_key:
        welcome_msg += "🚀 **Enhanced with premium AI** for the most accurate and detailed responses.\n\n"
    
    welcome_msg += """I can help you choose the perfect Apple Watch, compare models, find the best prices, and solve any technical issues.

**Try asking:**
• "Best Apple Watch for ₹30k budget"
• "Compare Apple Watch SE vs Series 9"  
• "My Apple Watch won't charge"
• "What's new in Apple Watch Series 9"
    """
    
    st.info(welcome_msg)

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(f"_{message['timestamp']}_")

# Chat Input
if prompt := st.chat_input("Ask me anything about Apple Watch..."):
    handle_user_input(prompt)

# Quick action buttons (only show when no conversation)
if not st.session_state.messages:
    st.markdown("### 🎯 **Popular Questions:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💰 Best watch for ₹25k budget"):
            handle_user_input("Best watch for ₹25k budget")
        
        if st.button("⚖️ Compare SE vs Series 9"):
            handle_user_input("Compare SE vs Series 9")
    
    with col2:
        if st.button("🔧 Watch won't charge"):
            handle_user_input("My Apple Watch won't charge")
            
        if st.button("❓ Series 9 features"):
            handle_user_input("What are Apple Watch Series 9 features")

# Professional footer
st.markdown("---")
footer_text = "Apple Watch Expert"
if bot.has_api_key:
    footer_text += " • Premium AI • Enhanced responses"
else:
    footer_text += " • Expert knowledge • Professional support"

footer_text += " • Current pricing"
st.caption(footer_text)