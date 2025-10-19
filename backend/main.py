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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3006"],
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
        prompt = f"""Categorize the topic/theme of the following text. Choose the most appropriate category.

Text: {text}

Categories: Politics, Economy, Security, Foreign Relations, Social Issues, Technology, Healthcare, Education, Environment, Defense, Trade, Culture, Infrastructure, Government Policy, Public Opinion

Respond with only the category name:"""
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

async def stream_llm_response(response) -> str:
    """Stream LLM response and return complete text"""
    complete_text = ""
    async for line in response.content:
        line_text = line.decode('utf-8').strip()
        if line_text.startswith('data: '):
            try:
                data = json.loads(line_text[6:])
                if 'choices' in data and len(data['choices']) > 0:
                    delta = data['choices'][0].get('delta', {})
                    if 'content' in delta:
                        complete_text += delta['content']
            except json.JSONDecodeError:
                continue
    return complete_text

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

def clean_data_for_json(data):
    """Recursively clean data to remove NaN values"""
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return 0.0
        return data
    elif data is None:
        return ""
    return data

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
async def analyze_sentiment(request: SentimentRequest, stream: bool = False):
    """Perform sentiment analysis on all rows"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[request.session_id]
    data = session_data.data
    
    # If streaming is requested, return streaming response
    if stream:
        return await stream_sentiment_analysis_impl(request.session_id, data)
    
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

async def stream_sentiment_analysis_impl(session_id: str, data: list):
    """Implementation for streaming sentiment analysis"""
    print(f"Stream analysis request for session: {session_id}")
    print(f"Found session with {len(data)} rows")
    
    session_data = sessions[session_id]
    
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
            "Access-Control-Allow-Origin": "http://localhost:3006",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@app.post("/mitigate")
async def generate_comments(request: CommentRequest, stream: bool = False):
    """Generate counter/amplify comments for analyzed data"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[request.session_id]
    analyzed_data = session_data.sentiments
    
    if not analyzed_data:
        raise HTTPException(status_code=400, detail="No analyzed data found")
    
    # If streaming is requested, return streaming response
    if stream:
        return await stream_comment_generation_impl(request.session_id, analyzed_data)
    
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

async def stream_comment_generation_impl(session_id: str, analyzed_data: list):
    """Implementation for streaming comment generation"""
    print(f"Stream comment generation request for session: {session_id}")
    
    session_data = sessions[session_id]
    
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
    
    async def generate_stream():
        final_data = []
        
        for i, row in enumerate(analyzed_data):
            text = str(row.get(text_column, ""))
            sentiment = row.get("sentiment", "neutral")
            
            # Send row start event
            yield f"data: {json.dumps({'type': 'row_start', 'index': i, 'total': len(analyzed_data), 'text': text[:100], 'sentiment': sentiment, 'topic': row.get('topic', 'General')})}\n\n"
            
            # Generate comment with streaming
            try:
                if sentiment == "negative":
                    response = await call_llm_api(text, "counter")
                elif sentiment == "positive":
                    response = await call_llm_api(text, "amplify")
                else:  # neutral
                    response = await call_llm_api(text, "neutral_positive")
                
                if hasattr(response, 'content'):  # Streaming response
                    complete_comment = ""
                    async for line in response.content:
                        line_text = line.decode('utf-8').strip()
                        if line_text.startswith('data: '):
                            try:
                                data = json.loads(line_text[6:])
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        word = delta['content']
                                        complete_comment += word
                                        # Send word-by-word update
                                        yield f"data: {json.dumps({'type': 'word', 'index': i, 'word': word, 'complete': complete_comment})}\n\n"
                            except json.JSONDecodeError:
                                continue
                    
                    final_comment = complete_comment
                else:
                    # Non-streaming response
                    final_comment = response.get("content", "No comment generated") if isinstance(response, dict) else str(response)
                    yield f"data: {json.dumps({'type': 'complete', 'index': i, 'comment': final_comment})}\n\n"
                
                # Add to final data
                final_row = {**row, "generated_comment": final_comment}
                final_data.append(final_row)
                
                # Send row complete event
                yield f"data: {json.dumps({'type': 'row_complete', 'index': i, 'comment': final_comment})}\n\n"
                
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
            "Access-Control-Allow-Origin": "http://localhost:3006",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    """Generate PowerPoint report"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[request.session_id]
    final_data = session_data.comments
    
    if not final_data:
        raise HTTPException(status_code=400, detail="No processed data found")
    
    # Create PowerPoint presentation
    prs = Presentation()
    
    # Title slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "Sentiment Analysis Report"
    subtitle.text = f"Malaysian External Intelligence Organisation (MEIO)\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nFile: {session_data.filename}"
    
    # Summary slide
    bullet_slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "Executive Summary"
    
    # Calculate statistics
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for row in final_data:
        sentiment_counts[row.get("sentiment", "neutral")] += 1
    
    total_rows = len(final_data)
    content = slide.placeholders[1]
    content.text = f"""Total Records Analyzed: {total_rows}
    
Sentiment Distribution:
• Positive: {sentiment_counts['positive']} ({sentiment_counts['positive']/total_rows*100:.1f}%)
• Negative: {sentiment_counts['negative']} ({sentiment_counts['negative']/total_rows*100:.1f}%)
• Neutral: {sentiment_counts['neutral']} ({sentiment_counts['neutral']/total_rows*100:.1f}%)

Analysis completed using advanced AI sentiment detection.
Mitigation strategies generated for all negative sentiments.
Amplification strategies provided for positive sentiments."""
    
    # Detailed analysis slides (first 10 entries)
    for i, row in enumerate(final_data[:10]):
        slide = prs.slides.add_slide(bullet_slide_layout)
        slide.shapes.title.text = f"Analysis #{i+1}"
        
        # Get text column
        text_content = ""
        for key, value in row.items():
            if key not in ["sentiment", "confidence", "generated_comment"] and isinstance(value, str) and len(str(value)) > 20:
                text_content = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                break
        
        content = slide.placeholders[1]
        content.text = f"""Original Text:
{text_content}

Detected Sentiment: {row.get('sentiment', 'Unknown').upper()}
Confidence: {row.get('confidence', 0):.2f}

Generated Response:
{row.get('generated_comment', 'No comment generated')[:300]}"""
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pptx')
    prs.save(temp_file.name)
    
    return FileResponse(
        temp_file.name,
        media_type='application/vnd.openxmlformats-officedocument.presentationml.presentation',
        filename=f"MEIO_Sentiment_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"
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

@app.get("/test-stream/{session_id}")
async def test_stream_endpoint(session_id: str):
    """Test streaming endpoint to verify routing"""
    return {"message": f"Test endpoint working for session {session_id}", "sessions": list(sessions.keys())}

@app.get("/simple-stream")
async def simple_stream_test():
    """Very simple streaming test"""
    async def generate():
        for i in range(5):
            yield f"data: {json.dumps({'message': f'Hello {i}'})}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")

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
                # Generate comment based on sentiment with tone matching
                if sentiment == "negative":
                    result = await call_llm_api(text, "counter")
                elif sentiment == "positive":
                    result = await call_llm_api(text, "amplify")
                else:  # neutral
                    result = await call_llm_api(text, "neutral_positive")
                
                print(f"LLM result type: {type(result)}, content: {str(result)[:100]}")
                
                # Handle different response types more robustly
                if hasattr(result, '__class__') and 'ClientResponse' in str(result.__class__):
                    # It's a response object, we need to extract content
                    final_comment = "Error: Received response object instead of content"
                elif isinstance(result, dict) and "content" in result:
                    final_comment = result["content"]
                elif isinstance(result, dict) and "error" in result:
                    final_comment = f"LLM Error: {result['error']}"
                elif isinstance(result, str):
                    final_comment = result
                else:
                    final_comment = f"Unexpected result type: {type(result)}"
                
                # Ensure we have valid comment text
                if not final_comment or final_comment.strip() == "" or "ClientResponse" in final_comment:
                    final_comment = f"Generated response for {sentiment} sentiment: This content requires further analysis."
                
                # Send word-by-word updates for streaming effect
                words = final_comment.split()
                complete_comment = ""
                for word in words:
                    complete_comment += word + " "
                    yield f"data: {json.dumps({'type': 'word', 'index': i, 'word': word + ' ', 'complete': complete_comment.strip()})}\n\n"
                    await asyncio.sleep(0.05)  # Faster word-by-word effect
                
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
            "Access-Control-Allow-Origin": "http://localhost:3006",
            "Access-Control-Allow-Credentials": "true"
        }
    )

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
            "Access-Control-Allow-Origin": "http://localhost:3006",
            "Access-Control-Allow-Credentials": "true"
        }
    )

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