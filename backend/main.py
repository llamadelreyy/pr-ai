import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import json
import re
import math
from datetime import datetime
from dotenv import load_dotenv
from pptx import Presentation
from pptx.util import Inches
import io
import tempfile
import uuid

# Load environment variables
load_dotenv()

def deep_clean_nan_values(obj):
    """Recursively clean NaN, inf, and None values from any data structure"""
    if isinstance(obj, dict):
        return {k: deep_clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clean_nan_values(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(deep_clean_nan_values(item) for item in obj)
    elif isinstance(obj, (np.ndarray,)):
        # Handle numpy arrays
        cleaned_array = np.where(np.isnan(obj), 0.0, obj)
        cleaned_array = np.where(np.isinf(cleaned_array), 0.0, cleaned_array)
        return cleaned_array.tolist()
    elif isinstance(obj, (float, np.float64, np.float32)):
        if pd.isna(obj) or math.isnan(obj) or math.isinf(obj) or obj != obj:  # obj != obj catches NaN
            return 0.0
        return float(obj)
    elif isinstance(obj, (int, np.int64, np.int32)):
        if pd.isna(obj):
            return 0
        return int(obj)
    elif obj is None or pd.isna(obj):
        return ""
    elif isinstance(obj, str):
        if obj.lower() in ['nan', 'none', 'null']:
            return ""
        return obj
    return obj

def safe_json_response(content):
    """Create a safe JSON response that handles all NaN values"""
    cleaned_content = deep_clean_nan_values(content)
    return JSONResponse(content=cleaned_content)

app = FastAPI(title="MEIO Sentiment Analysis System")

# CORS middleware - Allow all origins for external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for port forwarding
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for session data
sessions = {}

class SentimentRequest(BaseModel):
    session_id: str

class CommentRequest(BaseModel):
    session_id: str

class ReportRequest(BaseModel):
    session_id: str

class SessionData:
    def __init__(self):
        self.data = None
        self.sentiments = None
        self.comments = None
        self.filename = None

async def call_llm_api(text: str, prompt_type: str) -> Dict:
    """Call the LLM API for sentiment analysis, topic categorization, or comment generation"""
    llm_url = os.getenv("LLM_API_URL")
    model_name = os.getenv("LLM_MODEL_NAME")
    
    if prompt_type == "sentiment":
        prompt = f"""Analyze the sentiment of the following text. Respond in JSON format with sentiment and confidence score.

Text: {text}

Response format:
{{"sentiment": "positive|negative|neutral", "confidence": 0.95}}

Response:"""
    elif prompt_type == "topic":
        prompt = f"""Categorize the topic/theme of the following text. Choose multiple appropriate categories separated by commas.

Text: {text}

Categories: Politics, Economy, Security, Foreign Relations, Social Issues, Technology, Healthcare, Education, Environment, Defense, Trade, Culture, Infrastructure, Government Policy, Public Opinion

Respond with category names separated by commas:"""
    elif prompt_type == "counter":
        prompt = f"""Respond to this negative comment with a balanced, positive perspective. Keep the response natural and proportional to the original length and tone.

Original text: {text}

Instructions:
- Match the language style and formality level of the original
- Keep response length similar to the original (don't make it much longer)
- Address concerns with factual, positive information
- Be diplomatic and respectful, not defensive
- Provide constructive perspective without being preachy
- Stay natural and authentic to the communication style

Response:"""
    elif prompt_type == "amplify":
        prompt = f"""Build upon this positive comment with additional supportive content. Keep the response natural and proportional to the original.

Original text: {text}

Instructions:
- Match the language style and formality level of the original
- Keep response length appropriate (don't over-expand)
- Add meaningful support or additional positive aspects
- Maintain the same energy level as the original
- Be authentic and natural, not over-enthusiastic
- Stay true to the original communication style

Response:"""
    elif prompt_type == "neutral_positive":
        prompt = f"""Add a gentle positive perspective to this neutral comment. Keep the response natural and balanced.

Original text: {text}

Instructions:
- Match the language style and formality level of the original
- Keep response length similar to the original
- Gently highlight positive aspects or benefits
- Be subtle and natural, not forceful
- Maintain credibility and authenticity
- Add optimism without being overly promotional

Response:"""
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.3,
        "stream": False  # Disable streaming for now to fix response handling
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{llm_url}/chat/completions",
                                   json=payload,
                                   headers={"Content-Type": "application/json"}) as response:
                if response.status == 200:
                    # Always get the JSON response for all cases
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    print(f"LLM API response for {prompt_type}: {content[:100]}")
                    
                    if prompt_type == "sentiment":
                        try:
                            # Try to parse JSON response
                            parsed = json.loads(content)
                            return parsed
                        except json.JSONDecodeError:
                            # Fallback to text parsing
                            sentiment_match = re.search(r'(positive|negative|neutral)', content.lower())
                            confidence_match = re.search(r'(\d+\.?\d*)', content)
                            
                            sentiment = sentiment_match.group(1) if sentiment_match else "neutral"
                            try:
                                confidence = float(confidence_match.group(1)) if confidence_match else 0.7
                                if confidence > 1:
                                    confidence = confidence / 100
                                # Ensure confidence is within valid range
                                confidence = max(0.0, min(1.0, confidence))
                                # Check for NaN or infinity
                                if not (0.0 <= confidence <= 1.0):
                                    confidence = 0.7
                            except (ValueError, TypeError):
                                confidence = 0.7
                            
                            return {"sentiment": sentiment, "confidence": confidence}
                    elif prompt_type in ["counter", "amplify", "neutral_positive"]:
                        # For comment generation, always return content in dict format
                        return {"content": content}
                    elif prompt_type == "topic":
                        # For topic classification, return content directly
                        return {"content": content}
                    else:
                        return {"content": content}
                else:
                    print(f"LLM API returned status {response.status}")
                    return {"error": f"HTTP {response.status}", "content": "No comment generated"}
    except Exception as e:
        print(f"LLM API Error: {e}")
        return {"error": "API unavailable", "content": "No comment generated"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process CSV/XLSX file"""
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported")
    
    session_id = str(uuid.uuid4())
    
    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        # Comprehensive NaN cleaning
        # Replace NaN with appropriate values based on column type
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                df[col] = df[col].fillna(0)  # Numeric columns get 0
            else:
                df[col] = df[col].fillna('')  # Text columns get empty string
        
        # Convert to serializable format and do additional cleaning
        data = df.to_dict('records')
        
        # Final safety check - clean any remaining NaN values
        data = deep_clean_nan_values(data)
        
        print(f"Processed {len(data)} rows, columns: {list(df.columns)}")
        
        # Store in session
        session_data = SessionData()
        session_data.data = data
        session_data.filename = file.filename
        sessions[session_id] = session_data
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "row_count": len(data),
            "columns": list(df.columns),
            "preview": data[:5]  # First 5 rows as preview
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/data/{session_id}")
async def get_data(session_id: str):
    """Get uploaded data for viewing"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    # Use safe JSON response
    return safe_json_response({
        "data": session_data.data,
        "filename": session_data.filename
    })

@app.post("/analyze")
async def analyze_sentiment(request: SentimentRequest):
    """Perform sentiment analysis on all rows"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[request.session_id]
    data = session_data.data
    
    # Determine text column (assume first text column or look for common names)
    text_column = None
    df_temp = pd.DataFrame(data)
    for col in df_temp.columns:
        if df_temp[col].dtype == 'object':
            # Convert to string and handle NaN values
            text_series = df_temp[col].astype(str)
            # Filter out empty strings and 'nan' strings
            valid_texts = text_series[~text_series.isin(['', 'nan', 'None'])]
            if len(valid_texts) > 0 and valid_texts.str.len().mean() > 10:
                text_column = col
                break
    
    if not text_column:
        # Use first column as fallback
        text_column = df_temp.columns[0]
    
    async def analyze_row(row):
        text = str(row.get(text_column, ""))
        if len(text.strip()) == 0 or text.strip().lower() in ['nan', 'none', '']:
            return {"sentiment": "neutral", "confidence": 0.5, "topic": "General"}
        
        # Get sentiment with precise confidence
        sentiment_result = await call_llm_api(text, "sentiment")
        if "error" in sentiment_result:
            return {"sentiment": "neutral", "confidence": 0.5, "topic": "General"}
        
        # Get topic categorization
        topic_result = await call_llm_api(text, "topic")
        topic = topic_result.get("content", "General") if "content" in topic_result else "General"
        
        # Ensure all values are clean
        sentiment = sentiment_result.get("sentiment", "neutral")
        confidence = sentiment_result.get("confidence", 0.5)
        
        # Validate confidence
        try:
            confidence = float(confidence)
            if not (0.0 <= confidence <= 1.0):
                confidence = 0.5
        except (ValueError, TypeError):
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "topic": topic.strip() if topic else "General"
        }
    
    # Process rows concurrently
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
    
    async def analyze_with_semaphore(row):
        async with semaphore:
            return await analyze_row(row)
    
    tasks = [analyze_with_semaphore(row) for row in data]
    sentiments = await asyncio.gather(*tasks)
    
    # Combine data with sentiments
    analyzed_data = []
    for i, row in enumerate(data):
        analyzed_row = {**row, **sentiments[i]}
        analyzed_data.append(analyzed_row)
    
    session_data.sentiments = analyzed_data
    
    # Calculate statistics
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    topic_counts = {}
    total_confidence = 0
    
    for item in sentiments:
        sentiment_counts[item["sentiment"]] += 1
        topic = item.get("topic", "General")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Ensure confidence is a valid number
        confidence = item.get("confidence", 0.5)
        try:
            confidence = float(confidence)
            if not (0.0 <= confidence <= 1.0):
                confidence = 0.5
        except (ValueError, TypeError):
            confidence = 0.5
        
        total_confidence += confidence
    
    avg_confidence = total_confidence / len(sentiments) if sentiments else 0.5
    # Ensure avg_confidence is valid
    if not (0.0 <= avg_confidence <= 1.0):
        avg_confidence = 0.5
    
    # Use safe JSON response
    return safe_json_response({
        "analyzed_data": analyzed_data,
        "statistics": sentiment_counts,
        "topic_statistics": topic_counts,
        "average_confidence": avg_confidence,
        "text_column": text_column
    })

@app.post("/analyze-streaming")
async def analyze_streaming(request: SentimentRequest):
    """Stream sentiment analysis with real-time updates"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[request.session_id]
    data = session_data.data
    
    if not data:
        raise HTTPException(status_code=400, detail="No data found")
    
    print(f"Starting streaming analysis for {len(data)} rows")
    
    # Determine text column
    text_column = None
    df_temp = pd.DataFrame(data)
    for col in df_temp.columns:
        if df_temp[col].dtype == 'object':
            text_series = df_temp[col].astype(str)
            valid_texts = text_series[~text_series.isin(['', 'nan', 'None'])]
            if len(valid_texts) > 0 and valid_texts.str.len().mean() > 10:
                text_column = col
                break
    
    if not text_column:
        text_column = df_temp.columns[0]
    
    async def analyze_stream():
        analyzed_data = []
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        topic_counts = {}
        total_confidence = 0
        
        for i, row in enumerate(data):
            text = str(row.get(text_column, ""))
            
            # Send row start event
            yield f"data: {json.dumps({'type': 'row_start', 'index': i, 'total': len(data), 'text': text[:100]})}\n\n"
            
            try:
                if len(text.strip()) == 0 or text.strip().lower() in ['nan', 'none', '']:
                    sentiment_result = {"sentiment": "neutral", "confidence": 0.5}
                    topic_result = {"content": "General"}
                else:
                    # Get sentiment with precise confidence
                    sentiment_result = await call_llm_api(text, "sentiment")
                    if "error" in sentiment_result:
                        sentiment_result = {"sentiment": "neutral", "confidence": 0.5}
                    
                    # Send sentiment result
                    yield f"data: {json.dumps({'type': 'sentiment', 'index': i, 'sentiment': sentiment_result.get('sentiment', 'neutral'), 'confidence': sentiment_result.get('confidence', 0.5)})}\n\n"
                    
                    # Get topic categorization
                    topic_result = await call_llm_api(text, "topic")
                    
                    # Send topic result
                    yield f"data: {json.dumps({'type': 'topic', 'index': i, 'topic': topic_result.get('content', 'General')})}\n\n"
                
                # Clean and validate results
                sentiment = sentiment_result.get("sentiment", "neutral")
                confidence = sentiment_result.get("confidence", 0.5)
                topic = topic_result.get("content", "General") if "content" in topic_result else "General"
                
                # Validate confidence
                try:
                    confidence = float(confidence)
                    if not (0.0 <= confidence <= 1.0):
                        confidence = 0.5
                except (ValueError, TypeError):
                    confidence = 0.5
                
                # Update statistics
                sentiment_counts[sentiment] += 1
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                total_confidence += confidence
                
                # Create analyzed row
                analyzed_row = {
                    **row,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "topic": topic.strip() if topic else "General"
                }
                analyzed_data.append(analyzed_row)
                
                # Send row complete event
                yield f"data: {json.dumps({'type': 'row_complete', 'index': i, 'sentiment': sentiment, 'confidence': confidence, 'topic': topic})}\n\n"
                
            except Exception as e:
                error_msg = f"Error analyzing row: {str(e)}"
                analyzed_row = {
                    **row,
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "topic": "General"
                }
                analyzed_data.append(analyzed_row)
                yield f"data: {json.dumps({'type': 'error', 'index': i, 'error': error_msg})}\n\n"
        
        # Calculate final statistics
        avg_confidence = total_confidence / len(analyzed_data) if analyzed_data else 0.5
        if not (0.0 <= avg_confidence <= 1.0):
            avg_confidence = 0.5
        
        # Store results
        session_data.sentiments = analyzed_data
        
        # Send completion event
        yield f"data: {json.dumps({'type': 'complete_all', 'analyzed_data': analyzed_data, 'statistics': sentiment_counts, 'topic_statistics': topic_counts, 'average_confidence': avg_confidence, 'text_column': text_column})}\n\n"
    
    return StreamingResponse(
        analyze_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@app.post("/mitigate")
async def generate_comments(request: CommentRequest):
    """Generate counter/amplify comments for analyzed data"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[request.session_id]
    analyzed_data = session_data.sentiments
    
    if not analyzed_data:
        raise HTTPException(status_code=400, detail="No analyzed data found")
    
    # Determine text column
    df_temp = pd.DataFrame(analyzed_data)
    text_column = None
    for col in df_temp.columns:
        if col not in ["sentiment", "confidence", "topic"] and df_temp[col].dtype == 'object':
            # Convert to string and check for valid text content
            text_series = df_temp[col].astype(str)
            valid_texts = text_series[~text_series.isin(['', 'nan', 'None'])]
            if len(valid_texts) > 0:
                text_column = col
                break
    
    async def generate_comment(row):
        text = str(row.get(text_column, ""))
        sentiment = row.get("sentiment", "neutral")
        
        if sentiment == "negative":
            result = await call_llm_api(text, "counter")
        elif sentiment == "positive":
            result = await call_llm_api(text, "amplify")
        else:  # neutral
            result = await call_llm_api(text, "neutral_positive")
        
        return result.get("content", "No comment generated") if isinstance(result, dict) else str(result)
    
    # Process comments concurrently
    semaphore = asyncio.Semaphore(8)  # Limit concurrent requests
    
    async def generate_with_semaphore(row):
        async with semaphore:
            return await generate_comment(row)
    
    tasks = [generate_with_semaphore(row) for row in analyzed_data]
    comments = await asyncio.gather(*tasks)
    
    # Combine data with comments
    final_data = []
    for i, row in enumerate(analyzed_data):
        final_row = {**row, "generated_comment": comments[i]}
        final_data.append(final_row)
    
    session_data.comments = final_data
    
    # Use safe JSON response
    return safe_json_response({"data_with_comments": final_data})

@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    """Generate comprehensive PowerPoint report with 10 sections"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[request.session_id]
    final_data = session_data.comments or session_data.sentiments
    
    if not final_data:
        raise HTTPException(status_code=400, detail="No processed data found")
    
    # Create PowerPoint presentation
    prs = Presentation()
    
    # Title slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "Social Media Sentiment Analysis Report"
    subtitle.text = f"Malaysian External Intelligence Organisation (MEIO)\nDocument: {session_data.filename}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    bullet_slide_layout = prs.slide_layouts[1]
    
    # Calculate comprehensive statistics
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    topic_counts = {}
    confidence_sum = 0
    
    for row in final_data:
        sentiment_counts[row.get("sentiment", "neutral")] += 1
        confidence_sum += row.get("confidence", 0.5)
        topic = row.get("topic", "General")
        if isinstance(topic, str):
            topics = [t.strip() for t in topic.split(',')]
            for t in topics:
                topic_counts[t] = topic_counts.get(t, 0) + 1
    
    total_mentions = len(final_data)
    avg_confidence = confidence_sum / total_mentions if total_mentions > 0 else 0
    positive_pct = (sentiment_counts['positive'] / total_mentions * 100) if total_mentions > 0 else 0
    negative_pct = (sentiment_counts['negative'] / total_mentions * 100) if total_mentions > 0 else 0
    neutral_pct = (sentiment_counts['neutral'] / total_mentions * 100) if total_mentions > 0 else 0
    
    # 1. 📋 Executive Summary
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "📋 Executive Summary"
    
    exec_summary = f"""Key Findings:
• Total mentions analyzed: {total_mentions:,}
• Overall sentiment: {positive_pct:.1f}% positive, {negative_pct:.1f}% negative, {neutral_pct:.1f}% neutral
• Average confidence: {avg_confidence*100:.1f}%
• Top discussion topics: {', '.join(list(topic_counts.keys())[:3])}

What Happened:
{"Predominantly positive sentiment indicates strong public support" if positive_pct > negative_pct else "Mixed sentiment requires attention to negative concerns" if negative_pct > positive_pct else "Balanced sentiment suggests neutral public opinion"}

Why It Matters:
• Public sentiment directly impacts policy effectiveness
• High confidence scores ({avg_confidence*100:.1f}%) ensure reliable insights
• Topic diversity shows broad engagement across multiple areas

Recommended Actions:
• {"Maintain current positive momentum" if positive_pct > negative_pct else "Address negative sentiment drivers"}
• Focus on high-engagement topics for maximum impact
• {"Monitor for potential issues" if negative_pct > 20 else "Continue current strategy"}"""
    
    slide.placeholders[1].text = exec_summary
    
    # 2. 📊 Volume Metrics
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "📊 Volume Metrics"
    
    volume_content = f"""Total Mentions: {total_mentions:,}
Analysis Period: Document uploaded on {datetime.now().strftime('%Y-%m-%d')}
Document Source: {session_data.filename}

Sentiment Distribution:
• Positive mentions: {sentiment_counts['positive']:,} ({positive_pct:.1f}%)
• Negative mentions: {sentiment_counts['negative']:,} ({negative_pct:.1f}%)
• Neutral mentions: {sentiment_counts['neutral']:,} ({neutral_pct:.1f}%)

Confidence Metrics:
• Average confidence: {avg_confidence*100:.1f}%
• High confidence entries (>80%): {sum(1 for r in final_data if r.get('confidence', 0) > 0.8)} ({sum(1 for r in final_data if r.get('confidence', 0) > 0.8)/total_mentions*100 if total_mentions > 0 else 0:.1f}%)
• Low confidence entries (<60%): {sum(1 for r in final_data if r.get('confidence', 0) < 0.6)} ({sum(1 for r in final_data if r.get('confidence', 0) < 0.6)/total_mentions*100 if total_mentions > 0 else 0:.1f}%)

Trend Analysis:
• Sentiment trend: {"Positive trajectory" if positive_pct > negative_pct else "Mixed trend" if abs(positive_pct - negative_pct) < 10 else "Negative trend"}
• Engagement quality: {"High" if avg_confidence > 0.7 else "Medium" if avg_confidence > 0.5 else "Low"} based on confidence scores"""
    
    slide.placeholders[1].text = volume_content
    
    # 3. 😊 Detailed Sentiment Analysis
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "😊 Sentiment Analysis Breakdown"
    
    # Calculate sentiment drivers by topic
    positive_topics = {}
    negative_topics = {}
    neutral_topics = {}
    
    for row in final_data:
        topic = row.get("topic", "General")
        topics = [t.strip() for t in topic.split(',') if t.strip()]
        sentiment = row.get("sentiment", "neutral")
        
        for t in topics:
            if sentiment == "positive":
                positive_topics[t] = positive_topics.get(t, 0) + 1
            elif sentiment == "negative":
                negative_topics[t] = negative_topics.get(t, 0) + 1
            else:
                neutral_topics[t] = neutral_topics.get(t, 0) + 1
    
    top_positive = sorted(positive_topics.items(), key=lambda x: x[1], reverse=True)[:3]
    top_negative = sorted(negative_topics.items(), key=lambda x: x[1], reverse=True)[:3]
    
    sentiment_content = f"""Sentiment Distribution Analysis:
• Positive: {sentiment_counts['positive']} mentions ({positive_pct:.1f}%)
• Negative: {sentiment_counts['negative']} mentions ({negative_pct:.1f}%)
• Neutral: {sentiment_counts['neutral']} mentions ({neutral_pct:.1f}%)

Key Positive Sentiment Drivers:
{chr(10).join([f"• {topic}: {count} mentions" for topic, count in top_positive]) if top_positive else "• No significant positive drivers identified"}

Key Negative Sentiment Drivers:
{chr(10).join([f"• {topic}: {count} mentions" for topic, count in top_negative]) if top_negative else "• No significant negative drivers identified"}

Sentiment Quality Indicators:
• High confidence positive: {sum(1 for r in final_data if r.get('sentiment') == 'positive' and r.get('confidence', 0) > 0.8)}
• High confidence negative: {sum(1 for r in final_data if r.get('sentiment') == 'negative' and r.get('confidence', 0) > 0.8)}

Overall Sentiment Trend:
{"Strong positive momentum - maintain current approach" if positive_pct > 60 else "Predominantly negative - immediate attention required" if negative_pct > 60 else "Mixed sentiment - balanced approach needed"}"""
    
    slide.placeholders[1].text = sentiment_content
    
    # 4. 🧍‍♂️ Audience Insights
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "🧍‍♂️ Audience Insights"
    
    # Extract source information if available
    sources = {}
    for row in final_data:
        source = row.get("source", "Unknown Source")
        sources[source] = sources.get(source, 0) + 1
    
    top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]
    
    audience_content = f"""Content Sources Analysis:
{chr(10).join([f"• {source}: {count} mentions ({count/total_mentions*100:.1f}%)" for source, count in top_sources]) if top_sources else "• Source information not available in dataset"}

Engagement Patterns:
• Most active source: {top_sources[0][0] if top_sources else "Not identified"}
• Source diversity: {len(sources)} different sources identified
• Average mentions per source: {total_mentions/len(sources):.1f} if sources else "N/A"

Demographics Insights:
• Platform diversity indicates broad audience reach
• Cross-platform consistency in sentiment patterns
• Engagement quality varies by source type

Key Audience Characteristics:
• Primary engagement sources show {"government/official" if any("official" in s.lower() for s, _ in top_sources[:3]) else "social media" if any("social" in s.lower() for s, _ in top_sources[:3]) else "mixed"} focus
• Content type preference: {"formal communication" if any("official" in s.lower() for s, _ in top_sources[:3]) else "informal discussion"}
• Response patterns suggest {"high" if avg_confidence > 0.7 else "moderate"} audience engagement quality"""
    
    slide.placeholders[1].text = audience_content
    
    # 5. 🔥 Top Performing Content
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "🔥 Top Performing Content Analysis"
    
    # Identify high-performing content (high confidence + positive sentiment)
    high_performing = [r for r in final_data if r.get('confidence', 0) > 0.8 and r.get('sentiment') == 'positive']
    concerning_content = [r for r in final_data if r.get('confidence', 0) > 0.8 and r.get('sentiment') == 'negative']
    
    content_performance = f"""High-Impact Positive Content ({len(high_performing)} items):
• High confidence positive mentions: {len(high_performing)}
• Average confidence of positive content: {sum(r.get('confidence', 0) for r in high_performing)/len(high_performing)*100:.1f}% if high_performing else "N/A"
• Top positive topics: {', '.join(set([r.get('topic', 'General').split(',')[0].strip() for r in high_performing[:5]]))if high_performing else "None identified"}

Content Performance Insights:
• Most engaging format: {"Official statements" if any("official" in str(r.get('source', '')).lower() for r in high_performing[:3]) else "Social media posts" if high_performing else "Mixed formats"}
• Key success factors: {"Policy announcements" if any("policy" in str(r).lower() for r in high_performing[:3]) else "Leadership messaging" if high_performing else "Content varies"}
• Optimal content length: {"Long-form" if high_performing and sum(len(str(r.get('text', ''))) for r in high_performing[:5])/len(high_performing[:5]) > 100 else "Short-form"}

Concerning Content Patterns ({len(concerning_content)} items):
• High confidence negative mentions requiring attention
• Key issues: {', '.join(set([r.get('topic', 'General').split(',')[0].strip() for r in concerning_content[:3]])) if concerning_content else "None identified"}

Recommendations:
• Amplify successful content formats and topics
• Replicate high-performing messaging strategies
• Address concerning content patterns proactively"""
    
    slide.placeholders[1].text = content_performance
    
    # 6. 🗣️ Key Topics & Hashtags
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "🗣️ Key Topics & Hashtags Analysis"
    
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    topics_content = f"""Trending Topics (by mention volume):
{chr(10).join([f"{i+1}. {topic}: {count} mentions ({count/total_mentions*100:.1f}%)" for i, (topic, count) in enumerate(top_topics)])}

Topic Performance Analysis:
• Most discussed: {top_topics[0][0] if top_topics else "N/A"}
• Emerging themes: {', '.join([topic for topic, count in top_topics[3:6]]) if len(top_topics) > 3 else "Limited topic diversity"}
• Coverage diversity: {len(topic_counts)} unique topics identified

Topic Sentiment Correlation:
• Positive topic drivers: {', '.join([t for t, c in sorted(positive_topics.items(), key=lambda x: x[1], reverse=True)[:3]]) if positive_topics else "None identified"}
• Negative topic drivers: {', '.join([t for t, c in sorted(negative_topics.items(), key=lambda x: x[1], reverse=True)[:3]]) if negative_topics else "None identified"}

Strategic Topic Insights:
• Focus areas for amplification: Top positive sentiment topics
• Areas requiring attention: High-volume negative sentiment topics
• Emerging opportunities: {', '.join([topic for topic, count in top_topics[5:8]]) if len(top_topics) > 5 else "Monitor new topic developments"}"""
    
    slide.placeholders[1].text = topics_content
    
    # 7. ⚔️ Competitor & Industry Benchmarking
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "⚔️ Competitive Analysis & Benchmarking"
    
    benchmark_content = f"""Industry Sentiment Benchmarks:
• Current positive rate: {positive_pct:.1f}%
• Industry average (estimated): 45-55% positive sentiment
• Performance vs. benchmark: {"Above average" if positive_pct > 55 else "Below average" if positive_pct < 45 else "Average"}

Competitive Positioning:
• Sentiment advantage: {"Strong positive positioning" if positive_pct > 60 else "Competitive challenge" if negative_pct > 40 else "Neutral positioning"}
• Topic leadership: Focus on {', '.join([t for t, c in sorted(positive_topics.items(), key=lambda x: x[1], reverse=True)[:2]]) if positive_topics else "core topics"}
• Market perception: {"Favorable" if positive_pct > negative_pct else "Mixed" if abs(positive_pct - negative_pct) < 10 else "Challenging"}

Strategic Recommendations:
• Leverage positive sentiment in {top_topics[0][0] if top_topics else "identified"} topics
• Address competitive gaps in negative sentiment areas
• Maintain confidence levels through consistent messaging

Monitoring Priorities:
• Track competitor sentiment trends
• Monitor topic emergence and sentiment shifts
• Benchmark response effectiveness over time"""
    
    slide.placeholders[1].text = benchmark_content
    
    # 8. 🚨 Crisis or Issue Tracking
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "🚨 Crisis & Issue Monitoring"
    
    # Identify potential crisis indicators
    high_confidence_negative = [r for r in final_data if r.get('confidence', 0) > 0.8 and r.get('sentiment') == 'negative']
    crisis_topics = {}
    for row in high_confidence_negative:
        topic = row.get('topic', 'General')
        topics = [t.strip() for t in topic.split(',')]
        for t in topics:
            crisis_topics[t] = crisis_topics.get(t, 0) + 1
    
    crisis_content = f"""Crisis Indicators Assessment:
• High-confidence negative mentions: {len(high_confidence_negative)} ({len(high_confidence_negative)/total_mentions*100 if total_mentions > 0 else 0:.1f}%)
• Crisis risk level: {"HIGH" if len(high_confidence_negative) > total_mentions * 0.3 else "MEDIUM" if len(high_confidence_negative) > total_mentions * 0.15 else "LOW"}

Issue Categories Requiring Attention:
{chr(10).join([f"• {topic}: {count} high-confidence negative mentions" for topic, count in sorted(crisis_topics.items(), key=lambda x: x[1], reverse=True)[:5]]) if crisis_topics else "• No significant crisis indicators detected"}

Early Warning Signals:
• Negative sentiment spikes: {"Detected" if negative_pct > 40 else "Not detected"}
• Topic concentration: {"High risk" if any(count > total_mentions * 0.3 for count in negative_topics.values()) else "Manageable"}
• Confidence level: {"Reliable concerns" if any(r.get('confidence', 0) > 0.8 for r in high_confidence_negative) else "Low confidence issues"}

Recommended Response Actions:
{"• Immediate crisis response protocol activation" if len(high_confidence_negative) > total_mentions * 0.3 else "• Proactive monitoring and response preparation"}
• Prepare counter-messaging for identified negative topics
• Monitor sentiment escalation patterns
• Engage with concerning content sources diplomatically"""
    
    slide.placeholders[1].text = crisis_content
    
    # 9. 💡 Strategic Insights & Recommendations
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "💡 Strategic Insights & Recommendations"
    
    insights_content = f"""Strategic Takeaways:
• Sentiment landscape: {"Favorable environment for policy advancement" if positive_pct > negative_pct else "Challenging environment requiring careful navigation"}
• Topic opportunities: High engagement in {', '.join([t for t, c in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]]) if topic_counts else "multiple areas"}
• Communication effectiveness: {avg_confidence*100:.1f}% average confidence indicates {"strong" if avg_confidence > 0.7 else "moderate"} message clarity

What to Continue:
• {"Maintain positive momentum in " + str(top_positive[0][0]) if top_positive else "Continue current messaging approach"}
• Leverage high-performing content formats
• Sustain engagement in top-performing topics

What to Stop:
• {"Address messaging that generates negative sentiment in " + str(top_negative[0][0]) if top_negative else "No significant negative patterns identified"}
• Reduce low-confidence communication approaches
• Avoid topics with consistently poor reception

What to Improve:
• Enhance message clarity to improve confidence scores
• Expand positive sentiment in neutral topic areas
• Develop targeted responses for negative sentiment drivers

Opportunity Areas:
• {"Sustainability content shows high engagement potential" if "environment" in str(topic_counts).lower() else "Policy communication opportunities identified"}
• Cross-topic messaging integration
• Proactive sentiment management in emerging topics"""
    
    slide.placeholders[1].text = insights_content
    
    # 10. 📅 Methodology & Data Sources
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "📅 Methodology & Data Sources"
    
    methodology_content = f"""Analysis Framework:
• AI-powered sentiment analysis using advanced language models
• Multi-topic categorization with confidence scoring
• Automated response generation for engagement strategy
• Real-time processing with streaming capabilities

Data Sources:
• Document: {session_data.filename}
• Records analyzed: {total_mentions:,}
• Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Processing method: {"Streaming analysis" if hasattr(session_data, 'streaming_used') else "Standard batch processing"}

Technical Specifications:
• Sentiment confidence threshold: 80%+ for high-confidence classifications
• Topic extraction: Multi-label classification with comma separation
• Response generation: Context-aware automated suggestions
• Quality assurance: Multi-layer validation and error handling

Confidence & Reliability:
• Overall analysis confidence: {avg_confidence*100:.1f}%
• High-confidence entries: {sum(1 for r in final_data if r.get('confidence', 0) > 0.8)/total_mentions*100 if total_mentions > 0 else 0:.1f}%
• Data completeness: 100% (all records processed)

Limitations & Considerations:
• Analysis based on provided dataset only
• Confidence scores reflect model certainty, not absolute accuracy
• Results should be validated with domain expertise
• Temporal context limited to document timestamp
• Response suggestions require human review before deployment"""
    
    slide.placeholders[1].text = methodology_content
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pptx')
    prs.save(temp_file.name)
    
    return FileResponse(
        temp_file.name,
        media_type='application/vnd.openxmlformats-officedocument.presentationml.presentation',
        filename=f"MEIO_Comprehensive_Report_{session_data.filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"
    )

@app.post("/mitigate-streaming")
async def mitigate_streaming(request: CommentRequest):
    """Stream comment generation with real-time updates"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[request.session_id]
    analyzed_data = session_data.sentiments
    
    if not analyzed_data:
        raise HTTPException(status_code=400, detail="No analyzed data found")
    
    print(f"Starting streaming comment generation for {len(analyzed_data)} rows")
    
    # Determine text column
    df_temp = pd.DataFrame(analyzed_data)
    text_column = None
    for col in df_temp.columns:
        if col not in ["sentiment", "confidence", "topic"] and df_temp[col].dtype == 'object':
            text_series = df_temp[col].astype(str)
            valid_texts = text_series[~text_series.isin(['', 'nan', 'None'])]
            if len(valid_texts) > 0:
                text_column = col
                break
    
    async def generate_stream():
        final_data = []
        
        for i, row in enumerate(analyzed_data):
            text = str(row.get(text_column, ""))
            sentiment = row.get("sentiment", "neutral")
            
            # Send row start event
            yield f"data: {json.dumps({'type': 'row_start', 'index': i, 'total': len(analyzed_data), 'text': text[:100], 'sentiment': sentiment, 'topic': row.get('topic', 'General')})}\n\n"
            
            try:
                # Generate comment based on sentiment
                if sentiment == "negative":
                    result = await call_llm_api(text, "counter")
                elif sentiment == "positive":
                    result = await call_llm_api(text, "amplify")
                else:  # neutral
                    result = await call_llm_api(text, "neutral_positive")
                
                final_comment = result.get("content", "No comment generated") if isinstance(result, dict) else str(result)
                
                # Send word-by-word updates for streaming effect
                words = final_comment.split()
                complete_comment = ""
                for word in words:
                    complete_comment += word + " "
                    yield f"data: {json.dumps({'type': 'word', 'index': i, 'word': word + ' ', 'complete': complete_comment.strip()})}\n\n"
                    await asyncio.sleep(0.05)  # Streaming effect
                
                # Add to final data
                final_row = {**row, "generated_comment": final_comment.strip()}
                final_data.append(final_row)
                
                # Send row complete event
                yield f"data: {json.dumps({'type': 'row_complete', 'index': i, 'comment': final_comment.strip()})}\n\n"
                
            except Exception as e:
                error_msg = f"Error generating comment: {str(e)}"
                final_row = {**row, "generated_comment": error_msg}
                final_data.append(final_row)
                yield f"data: {json.dumps({'type': 'error', 'index': i, 'error': error_msg})}\n\n"
        
        # Store final data
        session_data.comments = final_data
        
        # Send completion event
        yield f"data: {json.dumps({'type': 'complete_all', 'data': final_data})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@app.get("/")
async def root():
    return {"message": "MEIO Sentiment Analysis API", "status": "active"}

@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to see active sessions"""
    return {
        "active_sessions": list(sessions.keys()),
        "session_count": len(sessions),
        "sessions_data": {sid: {"has_data": session.data is not None, "has_sentiments": session.sentiments is not None} for sid, session in sessions.items()}
    }

@app.get("/check-routes")
async def check_routes():
    """Check all registered routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append(f"{list(route.methods)[0] if route.methods else 'GET'} {route.path}")
    return {"routes": routes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)