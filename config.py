"""
API Configuration
Update your API keys here
"""

# ============================================================================
# OPENROUTER (Disabled)
# ============================================================================

# OpenRouter API Key (Disabled - Using Groq instead)
OPENROUTER_API_KEY = "YOUR_OPENROUTER_KEY_HERE"  # Disabled

# ============================================================================
# FREE OPEN-SOURCE MODELS (Backup options)
# ============================================================================

# DeepSeek API Key (Requires payment)
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_KEY_HERE"

# Hugging Face API Key (FREE - Many open-source models)
HUGGINGFACE_API_KEY = "YOUR_HF_KEY_HERE"

# Together AI API Key (FREE $25 credit)
TOGETHER_API_KEY = "YOUR_TOGETHER_KEY_HERE"

# ============================================================================
# PAID MODELS (Backup options)
# ============================================================================

# Gemini API Key (FREE tier available!)
# Get from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "YOUR_GEMINI_KEY_HERE"  # Add your Gemini API key here

# Groq API Keys (Multiple keys for maximum reliability!)
GROQ_API_KEYS = [
    "YOUR_GROQ_KEY_1_HERE",  # Key 1
    "YOUR_GROQ_KEY_2_HERE",  # Key 2
    "YOUR_GROQ_KEY_3_HERE"   # Key 3
]
GROQ_API_KEY = GROQ_API_KEYS[0]  # Primary key (for backward compatibility)

# ============================================================================
# MODEL PRIORITY (Which to try first)
# ============================================================================
# Groq (3 keys) → Gemini → Others (maximum reliability with Groq first!)
MODEL_PRIORITY = ["groq", "gemini"]  # Groq first with 3 keys, Gemini backup

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': ''
}
