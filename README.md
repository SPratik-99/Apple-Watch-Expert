# Apple Watch Expert - Web-Enabled AI System

A **comprehensive Apple Watch expert chatbot with real web scraping** that automatically fetches live data from apple.com when local information is insufficient, using intelligent AI model prioritization.

## 🌐 **Real Web Scraping Feature**

### **✅ How It Works:**
1. **Local Data First** ⚡ - Instant responses from your documents and knowledge base
2. **Web Scraping Fallback** 🔄 - **Actually scrapes apple.com** when local data doesn't have the answer
3. **Live Information** ✅ - Gets current pricing, latest features, new products

### **✅ What Gets Scraped:**
- **apple.com/apple-watch/** - Latest model information
- **apple.com/apple-watch/compare/** - Current model comparisons  
- **apple.com/apple-watch-series-9/** - Series 9 specific features
- **apple.com/apple-watch-se/** - SE current pricing and specs
- **apple.com/apple-watch-ultra-2/** - Ultra 2 adventure features
- **apple.com/apple-watch/health/** - Latest health monitoring capabilities

## 🧠 **AI Model Priority System**

### **✅ Automatic Model Selection (Priority Order):**

**🥇 Priority 1: Groq API** (if GROQ_API_KEY available)
- Tests `llama-3.3-70b-specdec` vs `llama-3.1-8b-instant`
- Selects best available automatically
- Fastest, most accurate responses
- 30 requests/minute free tier

**🥈 Priority 2: Hugging Face Transformers** (if torch installed)
- Local processing, complete privacy
- CPU-optimized models (DialoGPT-small)
- Works offline, no API limits
- Automatic fallback if Groq unavailable

**🥉 Priority 3: Ollama** (if running locally)
- Advanced local models (llama3.1:8b)
- Custom model configurations
- Full offline capability
- For power users with local setup

**🛡️ Priority 4: Expert Knowledge Base** (always available)
- Comprehensive Apple Watch knowledge
- Never fails, reliable fallback
- Expert-level responses even without AI models

## 🚀 **Setup Instructions**

### **1. Basic Installation**
```bash
# Install all dependencies (includes web scraping)
pip install -r requirements.txt
```

### **2. AI Model Setup (Recommended Priority Order)**

#### **🥇 Groq API (Priority 1 - Recommended)**
```bash
# 1. Get free API key from https://console.groq.com
# 2. Set environment variable
export GROQ_API_KEY="your_free_groq_api_key"

# Benefits:
# - System auto-tests llama-3.3-70b vs llama-3.1-8b
# - Selects best available model automatically
# - Fastest responses with highest accuracy
# - 30 requests/minute completely free
```

#### **🥈 Hugging Face (Priority 2 - Local Processing)**
```bash
# PyTorch and Transformers install automatically via requirements.txt
# Models download on first use (~500MB)

# Benefits:
# - Complete privacy - all processing local
# - No API limits or internet dependency  
# - CPU optimized for any hardware
# - Automatic fallback if Groq unavailable
```

#### **🥉 Ollama (Priority 3 - Advanced Users)**
```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Download model  
ollama pull llama3.1:8b

# Benefits:
# - Most powerful local models
# - Full customization and control
# - Advanced model configurations
# - Best for developers and power users
```

### **3. Setup Data Structure (Optional)**
```
data/
├── pdfs/           # Your Apple Watch PDF manuals (16 files)
├── txt/            # Your specification text files (2 files)  
└── prices/         # Your JSON pricing data (1 file)
```

### **4. Run the System**
```bash
streamlit run app.py
```

## 🎯 **Web Scraping Examples**

### **When Local Data is Sufficient:**
```
User: "Apple Watch SE price"
System: Uses local knowledge base → Instant response
Result: "Apple Watch SE: ₹24,900 (40mm), ₹28,900 (44mm)"
```

### **When Web Scraping is Triggered:**
```
User: "What's the latest Apple Watch news today"
System: Local data insufficient → Scrapes apple.com → Live response  
Result: "🔴 LIVE DATA from apple.com: [current information]"

User: "Current Apple Watch Series 9 features on apple.com"
System: Scrapes apple.com/apple-watch-series-9 → Fresh data
Result: "Based on live apple.com data: [latest features with timestamps]"
```

### **How You Know Data Source:**
- **Local Data**: Fast responses from knowledge base
- **Live Data**: Responses marked with "🔴 LIVE DATA from apple.com"
- **Cached Data**: Recent web data (refreshed every hour)

## 📋 **Complete File Structure**

```
apple-watch-expert/
├── 📄 app.py                           # Main application with web integration
├── 📄 config.py                        # Auto model testing and selection
├── 📄 web_enabled_multi_llm.py        # Multi-LLM with web scraping
├── 📄 real_apple_scraper.py           # Actual apple.com web scraper
├── 📄 document_loader.py              # Process your documents  
├── 📄 vector_store.py                 # ChromaDB embeddings
├── 📄 classifier.py                   # Input classification
├── 📄 requirements.txt                # All dependencies (updated)
├── 📄 README.md                       # This documentation
└── 📁 data/                           # Your optional data files
    ├── 📁 pdfs/                       # Apple Watch manuals
    ├── 📁 txt/                        # Text specifications  
    ├── 📁 prices/                     # JSON pricing data
    └── 📁 chroma_db/                  # Vector database
```

## 🔍 **Smart Data Flow**

```
User Question
    ↓
Question Classification
    ↓
Check Local Data First
    ↓
Sufficient? → YES → Use Local Data ⚡
    ↓
    NO → Scrape apple.com 🔄
    ↓
Combine Local + Web Data
    ↓
AI Model Selection (Groq → HF → Ollama → Knowledge)
    ↓
Expert Response with Data Source Indication
```

## 🎊 **Perfect Results**

### **✅ Web Scraping Working:**
- **"latest apple watch features"** → Scrapes apple.com for current info
- **"current pricing on apple website"** → Live pricing data  
- **"new apple watch models"** → Fresh product information
- **"apple watch series 9 today"** → Real-time feature updates

### **✅ Model Priority Working:**
- **With Groq API**: "🚀 Using Llama 3.3 70B - Most capable model"
- **Without Groq**: "🤖 Using Hugging Face - Local processing"  
- **With Ollama**: "🦙 Using Ollama - Advanced local models"
- **Fallback**: "🧠 Expert Mode - Knowledge base responses"

### **✅ Budget Questions Fixed:**
- **"best watch for 25k"** → Specific SE recommendation
- **"suggest watch in 40k budget"** → Series 9 detailed analysis
- **"I want under 30000"** → Perfect budget-aware advice

## 💡 **Key Advantages**

### **🌐 Always Current Information:**
✅ **Live Data**: Gets latest Apple Watch information when needed  
✅ **Current Pricing**: Real-time pricing from apple.com  
✅ **New Products**: Handles product launches not in local data  
✅ **Feature Updates**: Latest iOS/watchOS feature announcements  

### **🧠 Intelligent AI Priority:**
✅ **Best Model Selection**: Auto-tests and uses optimal AI model  
✅ **Graceful Fallbacks**: Never fails - comprehensive fallback chain  
✅ **Local Privacy**: Hugging Face option for complete privacy  
✅ **No Vendor Lock-in**: Uses multiple free AI sources  

### **⚡ Performance Optimized:**
✅ **Local First**: Instant responses from knowledge base when possible  
✅ **Smart Caching**: Web data cached for 1 hour to avoid repeated scraping  
✅ **CPU Optimized**: Works perfectly on any hardware  
✅ **Error Resilient**: Comprehensive error handling and recovery  

## 🔧 **Testing the System**

### **Test Web Scraping:**
```bash
streamlit run app.py

# Try these questions:
"What are the current Apple Watch prices on apple.com"
"Latest Apple Watch Series 9 features today"  
"New Apple Watch models announced recently"
```

### **Test Model Priority:**
```bash
# Without Groq API key:
export -n GROQ_API_KEY
streamlit run app.py  # Should use Hugging Face → Ollama → Knowledge

# With Groq API key:
export GROQ_API_KEY="your_key" 
streamlit run app.py  # Should use Groq (tests 3.3-70B vs 3.1-8B)
```

### **Verify Data Sources:**
- **Local responses**: Fast, no mention of apple.com
- **Web responses**: Include "🔴 LIVE DATA from apple.com"
- **Model used**: Shown in sidebar and status

## 🎉 **Result: Perfect Apple Watch Expert**

Your system now provides:

### **🌐 Real Web Integration:**
✅ **Actually scrapes apple.com** when local data insufficient  
✅ **Live pricing and features** from official Apple sources  
✅ **Current product information** for new launches  
✅ **Transparent data sourcing** - you know what's local vs web  

### **🧠 Optimal AI Performance:**
✅ **Correct priority**: Groq → Hugging Face → Ollama → Knowledge  
✅ **Auto model selection**: Tests 3.3-70B vs 3.1-8B automatically  
✅ **Universal compatibility**: Works on any hardware configuration  
✅ **Never fails**: Comprehensive fallback system  

### **💬 Expert Conversations:**
✅ **Perfect budget understanding**: Any format (₹20k, 25000, etc.)  
✅ **Context awareness**: Remembers conversation flow  
✅ **Expert troubleshooting**: Step-by-step technical solutions  
✅ **Current information**: Latest features and pricing  

**This is now a production-ready Apple Watch expert that actually uses live web data and optimal AI models! 🚀**
