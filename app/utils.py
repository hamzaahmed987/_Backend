import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import tweepy # type: ignore
from textblob import TextBlob # type: ignore
import re

# Load environment variables
load_dotenv()

# Configure APIs
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")
client = tweepy.Client(bearer_token=os.getenv("TWITTER_BEARER_TOKEN"))

def detect_language(text):
    """Language detect karta hai based on script and common words"""
    # Urdu/Arabic script check
    urdu_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
    # Hindi/Devanagari script check  
    hindi_pattern = r'[\u0900-\u097F]'
    
    # Common Urdu/Hindi words in Roman
    urdu_hindi_words = ['hai', 'hy', 'aur', 'ka', 'ki', 'ke', 'se', 'mein', 'kya', 'yeh', 'wo', 'hum', 'tum', 'ap', 'koi', 'sab']
    
    text_lower = text.lower()
    
    if re.search(urdu_pattern, text) or re.search(hindi_pattern, text):
        return 'urdu_hindi'
    elif any(word in text_lower for word in urdu_hindi_words):
        return 'urdu_hindi'
    else:
        return 'english'

def get_response_templates(lang):
    """Language-specific response templates with emojis"""
    if lang == 'urdu_hindi':
        return {
            'analyzing': "🔍 **News Ka Analysis Chal Raha Hai...** 📊",
            'ai_title': "🤖 **AI Expert Ki Detailed Ray:**",
            'sentiment_title': "💭 **Awaam Ka Overall Mood:**", 
            'tweets_title': "📱 **Log Kya Keh Rahe Hain:**",
            'verdict_title': "🚨 **FINAL JUDGMENT - YE KYA HAI?**",
            'real': "✅ **REAL NEWS HAI** - Bharosa kar sakte hain! 💯",
            'fake': "❌ **FAKE NEWS HAI** - Bilkul jhuta! ⚠️", 
            'propaganda': "⚠️ **PROPAGANDA HAI** - Kisi ka agenda push kar rahe! 🎭",
            'suspicious': "🤔 **SUSPICIOUS HAI** - Carefully dekho! 👀"
        }
    else:
        return {
            'analyzing': "🔍 **Analyzing News Content...** 📊",
            'ai_title': "🤖 **AI Expert Analysis:**",
            'sentiment_title': "💭 **Public Sentiment Overview:**",
            'tweets_title': "📱 **What People Are Saying:**", 
            'verdict_title': "🚨 **FINAL VERDICT - WHAT IS THIS?**",
            'real': "✅ **REAL NEWS** - You can trust this! 💯",
            'fake': "❌ **FAKE NEWS** - Completely false! ⚠️",
            'propaganda': "⚠️ **PROPAGANDA** - Someone's pushing agenda! 🎭", 
            'suspicious': "🤔 **SUSPICIOUS** - Be careful! 👀"
        }

def create_enhanced_prompt(text, lang):
    """Enhanced Gemini prompt with better instructions"""
    
    if lang == 'urdu_hindi':
        return f"""
🔍 **NEWS VERIFICATION ANALYSIS** 🔍

Ye news analyze karo detail mein: "{text}"

**ANALYSIS KARNEY KE LIYE:**
1. ✅ Factual accuracy check karo - kya ye sach hai?
2. 🔍 Sources ki credibility dekho - reliable hain ya nahi?
3. ⚠️ Propaganda patterns dhundo - koi agenda push kar rahe?
4. 📊 Misinformation signs check karo - typical fake news markers
5. 🎯 Emotional manipulation dekho - unnecessarily dramatic language?
6. 📰 Cross-verification - dusre sources mein same news hai?

**RESPONSE FORMAT (Urdu/Hindi mein attractive style mein):**
📋 **News Summary:** (2-3 lines mein kya keh rahi ye news)
🔍 **Detailed Analysis:** (Step by step reasoning with facts)
📚 **Sources Status:** (Reliable sources mention hain ya nahi)
🎯 **Red Flags:** (Agar koi suspicious cheezein hain)
⚖️ **Final Judgment:** (REAL/FAKE/PROPAGANDA/SUSPICIOUS - with confidence level)

**Response bilkul ek dost ki tarah simple language mein dena with emojis!**
"""
    else:
        return f"""
🔍 **NEWS VERIFICATION ANALYSIS** 🔍

Please analyze this news in detail: "{text}"

**ANALYSIS REQUIREMENTS:**
1. ✅ Check factual accuracy - is this actually true?
2. 🔍 Verify source credibility - are sources reliable?
3. ⚠️ Look for propaganda patterns - any agenda being pushed?
4. 📊 Check misinformation signs - typical fake news markers
5. 🎯 Spot emotional manipulation - unnecessarily dramatic language?
6. 📰 Cross-verification - does this appear in other reliable sources?

**RESPONSE FORMAT (in engaging conversational style):**
📋 **News Summary:** (2-3 lines about what this news claims)
🔍 **Detailed Analysis:** (Step by step reasoning with facts)
📚 **Sources Status:** (Are reliable sources mentioned or not)
🎯 **Red Flags:** (Any suspicious elements found)
⚖️ **Final Judgment:** (REAL/FAKE/PROPAGANDA/SUSPICIOUS - with confidence level)

**Keep response friendly and conversational with emojis!**
"""

def format_final_response(ai_response, sentiment_status, sample_tweets, lang):
    """Complete response formatting with all sections"""
    
    templates = get_response_templates(lang)
    
    # Clean AI response
    ai_clean = ai_response.replace('*', '').strip()
    
    # Sentiment with emoji
    sentiment_emojis = {
        'Positive': '😊 Positive (Logo khush lag rahe)' if lang == 'urdu_hindi' else '😊 Positive (People seem happy)',
        'Negative': '😠 Negative (Logo naraz/pareshaan)' if lang == 'urdu_hindi' else '😠 Negative (People seem upset)', 
        'Neutral': '😐 Neutral (Mixed reactions)' if lang == 'urdu_hindi' else '😐 Neutral (Mixed reactions)',
        'Unknown': '❓ Unknown (Data nahi mila)' if lang == 'urdu_hindi' else '❓ Unknown (No data available)'
    }
    
    # Build complete response
    response = f"""
{templates['analyzing']}

{templates['ai_title']}
{ai_clean}

{templates['sentiment_title']}
{sentiment_emojis.get(sentiment_status, sentiment_emojis['Unknown'])}

{templates['tweets_title']}"""
    
    # Add sample tweets (max 3, cleaned)
    tweet_count = 0
    for tweet in sample_tweets:
        if not any(error in tweet.lower() for error in ['error', 'exception']) and tweet_count < 3:
            cleaned_tweet = tweet[:120] + ('...' if len(tweet) > 120 else '')
            response += f"\n💬 {cleaned_tweet}"
            tweet_count += 1
    
    if tweet_count == 0:
        no_tweets_msg = "Koi tweets nahi mile is topic par 🤷‍♂️" if lang == 'urdu_hindi' else "No relevant tweets found 🤷‍♂️"
        response += f"\n💬 {no_tweets_msg}"
    
    return response

def determine_final_verdict(ai_response, sentiment_status, tweet_texts, lang):
    """Smart final verdict based on multiple factors"""
    
    templates = get_response_templates(lang)
    
    # AI response analysis
    ai_lower = ai_response.lower()
    
    # Keyword scoring
    fake_keywords = ['fake', 'false', 'misinformation', 'jhuta', 'galat', 'jhoot', 'untrue']
    propaganda_keywords = ['propaganda', 'biased', 'agenda', 'misleading', 'partial', 'one-sided']
    real_keywords = ['authentic', 'verified', 'credible', 'sach', 'real', 'true', 'confirmed']
    suspicious_keywords = ['suspicious', 'doubtful', 'unclear', 'shak', 'doubt']
    
    fake_score = sum(2 for keyword in fake_keywords if keyword in ai_lower)
    propaganda_score = sum(2 for keyword in propaganda_keywords if keyword in ai_lower) 
    real_score = sum(2 for keyword in real_keywords if keyword in ai_lower)
    suspicious_score = sum(1 for keyword in suspicious_keywords if keyword in ai_lower)
    
    # Tweet sentiment factor
    negative_tweet_count = sum(1 for tweet in tweet_texts 
                              if any(neg_word in tweet.lower() for neg_word in ['fake', 'false', 'lie', 'jhoot']))
    
    # Decision logic
    total_scores = fake_score + propaganda_score + real_score + suspicious_score
    
    if fake_score >= 2 or (fake_score > 0 and negative_tweet_count >= 2):
        verdict = templates['fake']
    elif propaganda_score >= 2:
        verdict = templates['propaganda'] 
    elif real_score >= 2 and fake_score == 0:
        verdict = templates['real']
    elif suspicious_score > 0 or total_scores == 0:
        verdict = templates['suspicious']
    else:
        verdict = templates['suspicious']  # Default to suspicious
    
    return f"\n{templates['verdict_title']}\n{verdict}"

def analyze_news(text):
    """Main enhanced analysis function"""
    
    # Detect language
    detected_lang = detect_language(text)
    
    # Create enhanced prompt
    enhanced_prompt = create_enhanced_prompt(text, detected_lang)
    
    # Gemini Analysis
    try:
        gemini_response = model.generate_content(enhanced_prompt)
        ai_opinion = getattr(gemini_response, "text", None)
        if not ai_opinion:
            ai_opinion = gemini_response.parts[0].text if gemini_response.parts else "No AI response."
    except Exception as e:
        error_msg = f"AI Analysis mein problem: {e}" if detected_lang == 'urdu_hindi' else f"AI Analysis error: {e}"
        ai_opinion = error_msg

    # Twitter Analysis  
    try:
        tweets = client.search_recent_tweets(query=text, max_results=10)
        tweet_texts = [t.text for t in tweets.data] if tweets and tweets.data else []
    except Exception as e:
        error_msg = f"Twitter data nahi mila: {e}" if detected_lang == 'urdu_hindi' else f"Twitter error: {e}"
        tweet_texts = [error_msg]

    # Sentiment Analysis
    try:
        valid_tweets = [t for t in tweet_texts if not any(err in t.lower() for err in ['error', 'nahi mila'])]
        if valid_tweets:
            sentiments = [TextBlob(t).sentiment.polarity for t in valid_tweets]
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            if avg_sentiment > 0.1:
                sentiment_status = "Positive"
            elif avg_sentiment < -0.1:
                sentiment_status = "Negative" 
            else:
                sentiment_status = "Neutral"
        else:
            sentiment_status = "Unknown"
    except:
        sentiment_status = "Unknown"

    # Format complete response
    formatted_response = format_final_response(ai_opinion, sentiment_status, tweet_texts, detected_lang)
    
    # Add final verdict
    final_verdict = determine_final_verdict(ai_opinion, sentiment_status, tweet_texts, detected_lang)
    complete_response = formatted_response + final_verdict
    
    return {
        "complete_response": complete_response,
        "ai_analysis": ai_opinion,
        "public_sentiment": sentiment_status, 
        "sample_tweets": tweet_texts,
        "detected_language": detected_lang
    }

# Usage example
if __name__ == "__main__":
    # Test kar sakte ho
    news_input = input("News enter karo (Urdu/Hindi/English): ")
    result = analyze_news(news_input)
    print(result["complete_response"])






















































# -------------------------\




























# import os
# import asyncio
# import sqlite3
# import hashlib
# import json
# from datetime import datetime, timedelta
# from typing import Dict, List, Optional, Tuple
# from dataclasses import dataclass
# from enum import Enum
# import logging
# import time
# import random

# # FREE imports only
# import requests
# from textblob import TextBlob
# import feedparser
# import re
# import numpy as np
# from collections import Counter, defaultdict
# import warnings
# warnings.filterwarnings("ignore")

# # FREE language detection
# try:
#     from langdetect import detect, detect_langs, LangDetectException
# except ImportError:
#     print("Installing langdetect...")
#     os.system("pip install langdetect")
#     from langdetect import detect, detect_langs, LangDetectException

# # FREE web scraping
# try:
#     from bs4 import BeautifulSoup
# except ImportError:
#     print("Installing beautifulsoup4...")
#     os.system("pip install beautifulsoup4")
#     from bs4 import BeautifulSoup

# # FREE news scraping
# try:
#     from newspaper import Article
# except ImportError:
#     print("Installing newspaper3k...")
#     os.system("pip install newspaper3k")
#     from newspaper import Article

# # Logging setup
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @dataclass
# class NewsAnalysis:
#     """News analysis result structure"""
#     original_text: str
#     detected_language: str
#     confidence_score: float
#     classification: str
#     analysis_text: str
#     social_sentiment: Dict[str, float]
#     cross_references: List[Dict]
#     credibility_score: float
#     response_text: str
#     metadata: Dict

# class NewsClassification(Enum):
#     """News classification categories"""
#     VERIFIED_TRUE = "verified_true"
#     LIKELY_TRUE = "likely_true"
#     UNCERTAIN = "uncertain"
#     LIKELY_FALSE = "likely_false"
#     VERIFIED_FALSE = "verified_false"
#     PROPAGANDA = "propaganda"
#     SATIRE = "satire"
#     OPINION = "opinion"
#     OUTDATED = "outdated"

# class FreeNewsVerifier:
#     """100% FREE Advanced Global News Verification System"""
    
#     def __init__(self):
#         """Initialize with FREE services only"""
#         self.setup_free_services()
#         self.setup_database()
#         self.setup_language_support()
#         self.load_free_sources()
        
#     def setup_free_services(self):
#         """Setup completely FREE services"""
#         # Free RSS feeds for news
#         self.free_rss_feeds = {
#             'international': [
#                 'http://feeds.bbci.co.uk/news/rss.xml',
#                 'https://rss.cnn.com/rss/edition.rss',
#                 'https://feeds.reuters.com/reuters/topNews',
#                 'https://feeds.npr.org/1001/rss.xml'
#             ],
#             'pakistan': [
#                 'https://www.dawn.com/feeds/home',
#                 'https://www.thenews.com.pk/rss/1/1',
#                 'https://tribune.com.pk/feed/home'
#             ],
#             'india': [
#                 'https://www.thehindu.com/feeder/default.rss',
#                 'https://indianexpress.com/feed/',
#                 'https://www.hindustantimes.com/feeds/rss/news/rssfeed.xml'
#             ],
#             'arabic': [
#                 'https://www.aljazeera.net/aljazeerarss/a7c186be-1baa-4bd4-9d85-710b2b2fcfb6/73d0e1b4-532f-45ef-b135-a124d82e8393',
#                 'https://arabic.rt.com/rss'
#             ]
#         }
        
#         # Free fact-checking websites
#         self.fact_check_sites = [
#             'https://www.snopes.com',
#             'https://www.factcheck.org',
#             'https://www.politifact.com',
#             'https://fullfact.org',
#             'https://www.boomlive.in',
#             'https://www.soch.pk'
#         ]
        
#     def setup_database(self):
#         """Setup SQLite database (completely free)"""
#         self.db_path = "free_news_verification.db"
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()
        
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS news_analyses (
#                 id INTEGER PRIMARY KEY,
#                 text_hash TEXT UNIQUE,
#                 original_text TEXT,
#                 language TEXT,
#                 classification TEXT,
#                 confidence_score REAL,
#                 analysis_data TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#         """)
        
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS free_sources (
#                 id INTEGER PRIMARY KEY,
#                 domain TEXT UNIQUE,
#                 credibility_score REAL,
#                 bias_score REAL,
#                 category TEXT,
#                 language TEXT
#             )
#         """)
        
#         conn.commit()
#         conn.close()
        
#     def setup_language_support(self):
#         """Setup comprehensive FREE language support"""
#         self.language_configs = {
#             'en': {
#                 'name': 'English',
#                 'rtl': False,
#                 'greetings': ['Hey there!', 'Check this out!', 'What\'s up!'],
#                 'exclamations': ['Wow!', 'No way!', 'That\'s crazy!']
#             },
#             'ur': {
#                 'name': 'اردو',
#                 'rtl': True,
#                 'greetings': ['السلام علیکم!', 'دیکھیں یہ!', 'کیا حال ہے!'],
#                 'exclamations': ['واہ!', 'نہیں یار!', 'کمال ہے!']
#             },
#             'hi': {
#                 'name': 'हिंदी',
#                 'rtl': False,
#                 'greetings': ['नमस्ते!', 'देखिए यह!', 'क्या हाल है!'],
#                 'exclamations': ['वाह!', 'नहीं यार!', 'कमाल है!']
#             },
#             'ar': {
#                 'name': 'العربية',
#                 'rtl': True,
#                 'greetings': ['السلام عليكم!', 'انظر هذا!', 'أهلاً!'],
#                 'exclamations': ['رائع!', 'لا يمكن!', 'مذهل!']
#             }
#         }
        
#     def load_free_sources(self):
#         """Load FREE credible sources database"""
#         free_credible_sources = {
#             'bbc.com': {'credibility': 0.9, 'bias': 0.1, 'category': 'news', 'lang': 'en'},
#             'reuters.com': {'credibility': 0.95, 'bias': 0.05, 'category': 'news', 'lang': 'en'},
#             'dawn.com': {'credibility': 0.85, 'bias': 0.15, 'category': 'news', 'lang': 'ur'},
#             'thehindu.com': {'credibility': 0.88, 'bias': 0.12, 'category': 'news', 'lang': 'hi'},
#             'aljazeera.com': {'credibility': 0.82, 'bias': 0.20, 'category': 'news', 'lang': 'en'},
#             'infowars.com': {'credibility': 0.1, 'bias': 0.9, 'category': 'conspiracy', 'lang': 'en'},
#         }
        
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()
        
#         for domain, info in free_credible_sources.items():
#             cursor.execute("""
#                 INSERT OR REPLACE INTO free_sources 
#                 (domain, credibility_score, bias_score, category, language) 
#                 VALUES (?, ?, ?, ?, ?)
#             """, (domain, info['credibility'], info['bias'], info['category'], info['lang']))
        
#         conn.commit()
#         conn.close()
    
#     def detect_language_free(self, text: str) -> Tuple[str, float]:
#         """FREE language detection"""
#         try:
#             detected_langs = detect_langs(text)
#             if detected_langs:
#                 lang_code = detected_langs[0].lang
#                 confidence = detected_langs[0].prob
                
#                 lang_mapping = {'hi': 'hi', 'ur': 'ur', 'en': 'en', 'ar': 'ar', 'fa': 'ur', 'bn': 'hi'}
#                 final_lang = lang_mapping.get(lang_code, 'en')
#                 return final_lang, confidence
#         except:
#             pass
            
#         return self.detect_language_by_script(text)
    
#     def detect_language_by_script(self, text: str) -> Tuple[str, float]:
#         """FREE script-based language detection"""
#         urdu_arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F]'  
#         hindi_pattern = r'[\u0900-\u097F]'
        
#         urdu_words = ['ہے', 'اور', 'کا', 'میں', 'hai', 'aur', 'ka']
#         hindi_words = ['है', 'और', 'का', 'में']
#         arabic_words = ['في', 'من', 'على', 'ما']
        
#         if re.search(urdu_arabic_pattern, text):
#             arabic_count = sum(1 for word in arabic_words if word in text)
#             urdu_count = sum(1 for word in urdu_words if word in text.lower())
#             return ('ar', 0.8) if arabic_count > urdu_count else ('ur', 0.8)
#         elif re.search(hindi_pattern, text):
#             return 'hi', 0.9
#         elif any(word in text.lower() for word in urdu_words):
#             return 'ur', 0.7
#         elif any(word in text for word in hindi_words):
#             return 'hi', 0.7
        
#         return 'en', 0.6
    
#     def detect_fake_indicators(self, text: str) -> List[str]:
#         """FREE fake news indicator detection"""
#         indicators = []
#         text_lower = text.lower()
        
#         sensational_words = ['shocking', 'unbelievable', 'breaking', 'urgent', 'you won\'t believe']
#         if any(word in text_lower for word in sensational_words):
#             indicators.append('sensational_language')
        
#         if len(re.findall(r'[A-Z]{3,}', text)) > 2:
#             indicators.append('excessive_caps')
        
#         vague_phrases = ['some people say', 'sources claim', 'reportedly']
#         if any(phrase in text_lower for phrase in vague_phrases):
#             indicators.append('vague_sourcing')
        
#         conspiracy_words = ['cover up', 'hidden truth', 'conspiracy']
#         if any(word in text_lower for word in conspiracy_words):
#             indicators.append('conspiracy_language')
        
#         return indicators
    
#     def analyze_mentioned_sources(self, text: str) -> Dict[str, bool]:
#         """FREE source analysis"""
#         urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', text)
#         source_patterns = [r'according to (\w+)', r'(\w+) reported']
        
#         mentioned_sources = []
#         for pattern in source_patterns:
#             matches = re.findall(pattern, text, re.IGNORECASE)
#             mentioned_sources.extend(matches)
        
#         has_sources = len(urls) > 0 or len(mentioned_sources) > 0
#         credible = False
        
#         if mentioned_sources:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
#             for source in mentioned_sources:
#                 cursor.execute("SELECT credibility_score FROM free_sources WHERE domain LIKE ?", (f'%{source.lower()}%',))
#                 result = cursor.fetchone()
#                 if result and result[0] > 0.7:
#                     credible = True
#                     break
#             conn.close()
        
#         return {'has_sources': has_sources, 'credible': credible, 'source_count': len(mentioned_sources) + len(urls)}
    
#     def free_ai_analysis(self, text: str, language: str) -> str:
#         """FREE AI-like analysis"""
#         fake_indicators = self.detect_fake_indicators(text)
#         source_analysis = self.analyze_mentioned_sources(text)
        
#         emotional_words = ['shocking', 'terrible', 'amazing', 'incredible']
#         emotional_count = sum(1 for word in emotional_words if word in text.lower())
#         high_emotion = emotional_count > len(text.split()) * 0.05
        
#         numbers = re.findall(r'\b\d+\b', text)
#         dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
#         verifiable = len(numbers) + len(dates) > 2
        
#         confidence = max(0, min(100, 50 - len(fake_indicators) * 10 + (15 if source_analysis['has_sources'] else 0) + (20 if source_analysis['credible'] else 0) + (15 if verifiable else 0)))
        
#         if language in ['ur', 'hi']:
#             analysis = f"""
# 📋 **خبر کا تجزیہ:** یہ خبر کی جانچ مکمل ہو گئی ہے۔

# 🔍 **نتائج:**
# • **جذباتی زبان:** {'زیادہ جذباتی' if high_emotion else 'متوازن'}
# • **ذرائع:** {'موجود' if source_analysis['has_sources'] else 'غیر موجود'}
# • **حقائق:** {'قابل تصدیق' if verifiable else 'مبہم'}

# 🎯 **خطرے کے اشارے:** {len(fake_indicators)} ملے
# ⚖️ **اعتماد:** {confidence}%
# """
#         else:
#             analysis = f"""
# 📋 **News Analysis:** Verification complete.

# 🔍 **Results:**
# • **Emotional Language:** {'High' if high_emotion else 'Balanced'}
# • **Sources:** {'Present' if source_analysis['has_sources'] else 'Missing'}
# • **Facts:** {'Verifiable' if verifiable else 'Vague'}

# 🎯 **Red Flags:** {len(fake_indicators)} detected
# ⚖️ **Confidence:** {confidence}%
# """
        
#         return analysis
    
#     async def free_social_sentiment(self, text: str) -> Dict[str, float]:
#         """FREE sentiment analysis"""
#         try:
#             blob = TextBlob(text)
#             sentiment_score = blob.sentiment.polarity
            
#             return {
#                 'overall_sentiment': sentiment_score,
#                 'confidence': abs(sentiment_score),
#                 'positive_ratio': 0.6 if sentiment_score > 0.1 else 0.2,
#                 'negative_ratio': 0.2 if sentiment_score > 0.1 else 0.6 if sentiment_score < -0.1 else 0.3,
#                 'neutral_ratio': 0.2 if abs(sentiment_score) > 0.1 else 0.5,
#                 'sample_count': 1
#             }
#         except:
#             return {'overall_sentiment': 0.0, 'confidence': 0.5, 'positive_ratio': 0.33, 'negative_ratio': 0.33, 'neutral_ratio': 0.34, 'sample_count': 0}
    
#     async def free_cross_reference(self, text: str, language: str) -> List[Dict]:
#         """FREE cross-reference using RSS feeds"""
#         references = []
        
#         try:
#             keywords = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())[:3]
#             relevant_feeds = self.free_rss_feeds.get('international', [])
            
#             if language == 'ur':
#                 relevant_feeds.extend(self.free_rss_feeds.get('pakistan', []))
#             elif language == 'hi':
#                 relevant_feeds.extend(self.free_rss_feeds.get('india', []))
            
#             for feed_url in relevant_feeds[:3]:
#                 try:
#                     feed = feedparser.parse(feed_url)
#                     for entry in feed.entries[:5]:
#                         title = entry.get('title', '').lower()
#                         if any(keyword in title for keyword in keywords):
#                             references.append({
#                                 'title': entry.get('title', 'No title'),
#                                 'url': entry.get('link', ''),
#                                 'source': feed_url,
#                                 'relevance_score': 0.7
#                             })
#                 except:
#                     continue
#         except:
#             pass
        
#         return references[:5]
    
#     def determine_classification(self, fake_indicators: List[str], source_analysis: Dict, confidence: float) -> NewsClassification:
#         """Determine news classification"""
#         if confidence >= 80:
#             return NewsClassification.VERIFIED_TRUE
#         elif confidence >= 60:
#             return NewsClassification.LIKELY_TRUE
#         elif confidence >= 40:
#             return NewsClassification.UNCERTAIN
#         elif confidence >= 20:
#             return NewsClassification.LIKELY_FALSE
#         else:
#             return NewsClassification.VERIFIED_FALSE
    
#     def create_social_response(self, analysis: NewsAnalysis) -> str:
#         """Create engaging social media response"""
#         lang_config = self.language_configs.get(analysis.detected_language, self.language_configs['en'])
#         greeting = random.choice(lang_config['greetings'])
        
#         if analysis.detected_language in ['ur', 'hi']:
#             return f"""
# {greeting} 

# 🔍 **خبر کی تصدیق مکمل!**

# 📊 **نتیجہ:** {analysis.credibility_score}% اعتماد
# 🎯 **درجہ:** {analysis.classification.value.replace('_', ' ').title()}

# {analysis.analysis_text}

# 📚 **متعلقہ خبریں:** {len(analysis.cross_references)} ملیں
# 💭 **عوامی رائے:** {analysis.social_sentiment.get('overall_sentiment', 0):.1f}

# #خبر_کی_تصدیق #حقائق_کی_جانچ
# """
#         else:
#             return f"""
# {greeting}

# 🔍 **News Verification Complete!**

# 📊 **Result:** {analysis.credibility_score}% Confidence  
# 🎯 **Classification:** {analysis.classification.value.replace('_', ' ').title()}

# {analysis.analysis_text}

# 📚 **Related Articles:** {len(analysis.cross_references)} found
# 💭 **Public Sentiment:** {analysis.social_sentiment.get('overall_sentiment', 0):.1f}

# #NewsVerification #FactCheck
# """
    
#     async def verify_news(self, news_text: str, source_url: Optional[str] = None) -> NewsAnalysis:
#         """Main verification function"""
#         try:
#             # Language detection
#             language, lang_confidence = self.detect_language_free(news_text)
            
#             # Generate hash for caching
#             text_hash = hashlib.md5(news_text.encode()).hexdigest()
            
#             # Check cache first
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
#             cursor.execute("SELECT * FROM news_analyses WHERE text_hash = ?", (text_hash,))
#             cached_result = cursor.fetchone()
#             conn.close()
            
#             if cached_result:
#                 logger.info("Using cached analysis")
#                 analysis_data = json.loads(cached_result[5])
#                 return NewsAnalysis(**analysis_data)
            
#             # Perform analysis
#             analysis_text = self.free_ai_analysis(news_text, language)
#             social_sentiment = await self.free_social_sentiment(news_text)
#             cross_references = await self.free_cross_reference(news_text, language)
            
#             # Calculate scores
#             fake_indicators = self.detect_fake_indicators(news_text)
#             source_analysis = self.analyze_mentioned_sources(news_text)
#             credibility_score = max(0, min(100, 50 - len(fake_indicators) * 10 + (15 if source_analysis['has_sources'] else 0) + (20 if source_analysis['credible'] else 0)))
            
#             classification = self.determine_classification(fake_indicators, source_analysis, credibility_score)
            
#             # Create analysis object
#             analysis = NewsAnalysis(
#                 original_text=news_text,
#                 detected_language=language,
#                 confidence_score=lang_confidence,
#                 classification=classification.value,
#                 analysis_text=analysis_text,
#                 social_sentiment=social_sentiment,
#                 cross_references=cross_references,
#                 credibility_score=credibility_score,
#                 response_text="",
#                 metadata={
#                     'fake_indicators': fake_indicators,
#                     'source_analysis': source_analysis,
#                     'processing_time': time.time(),
#                     'source_url': source_url
#                 }
#             )
            
#             # Generate social response
#             analysis.response_text = self.create_social_response(analysis)
            
#             # Cache result
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT OR REPLACE INTO news_analyses 
#                 (text_hash, original_text, language, classification, confidence_score, analysis_data) 
#                 VALUES (?, ?, ?, ?, ?, ?)
#             """, (text_hash, news_text, language, classification.value, credibility_score, json.dumps(analysis.__dict__, default=str)))
#             conn.commit()
#             conn.close()
            
#             return analysis
            
#         except Exception as e:
#             logger.error(f"Verification failed: {e}")
#             raise

# # Usage Example
# async def main():
#     """Example usage"""
#     verifier = FreeNewsVerifier()
    
#     # Test news in different languages
#     test_news = [
#         "Breaking: Scientists discover revolutionary cure that doctors don't want you to know!",
#         "وزیر اعظم نے آج پارلیمنٹ میں اہم اعلان کیا۔ ذرائع کے مطابق یہ معاشی پالیسی سے متعلق ہے۔",
#         "प्रधानमंत्री ने आज संसद में महत्वपूर्ण घोषणा की। सूत्रों के अनुसार यह आर्थिक नीति से संबंधित है।"
#     ]
    
#     for news in test_news:
#         print(f"\n{'='*50}")
#         print(f"Testing: {news[:50]}...")
        
#         try:
#             analysis = await verifier.verify_news(news)
#             print(analysis.response_text)
#         except Exception as e:
#             print(f"Error: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())