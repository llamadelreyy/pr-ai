# MEIO Sentiment Analysis System

🇲🇾 **Malaysian External Intelligence Organisation (MEIO)**  
Advanced Sentiment Analysis and Mitigation Platform

## Overview

The MEIO Sentiment Analysis System is a comprehensive platform designed for government intelligence operations to analyze public sentiment, identify potential threats, and generate strategic responses. The system processes large datasets (up to 13,000 rows) efficiently using AI-powered sentiment analysis and generates actionable intelligence reports.

## Features

### 🔍 **Multi-Step Analysis Workflow**
1. **File Upload** - Support for CSV and XLSX files
2. **Data Visualization** - Interactive table view of uploaded data
3. **AI Sentiment Analysis** - Real-time sentiment classification (Positive/Negative/Neutral)
4. **Strategic Mitigation** - Automated generation of counter-narratives and amplification strategies
5. **Intelligence Reports** - Comprehensive PowerPoint presentations for decision makers

### 🚀 **Performance & Scalability**
- **Concurrent Processing** - Handles up to 13,000 rows with parallel AI analysis
- **Real-time Updates** - Live progress indicators and statistics
- **Optimized Backend** - FastAPI with async processing for maximum throughput

### 🎨 **Professional Interface**
- **Government-Grade Design** - Formal, secure, and professional UI
- **MEIO Branding** - Custom styled for Malaysian intelligence operations
- **Responsive Layout** - Works on desktop and tablet devices
- **Accessibility** - WCAG compliant design patterns

### 🤖 **AI Integration**
- **External LLM Support** - Configurable AI model endpoint
- **Advanced Prompting** - Specialized prompts for intelligence analysis
- **Multi-Strategy Response** - Different approaches for positive, negative, and neutral sentiments

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │   External LLM  │
│                 │    │                  │    │                 │
│ • File Upload   │◄──►│ • Data Processing│◄──►│ • Sentiment AI  │
│ • Data Tables   │    │ • Async Workers  │    │ • Comment Gen   │
│ • Progress UI   │    │ • Session Mgmt   │    │ • Strategic AI  │
│ • Report DL     │    │ • Report Gen     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites
- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **LLM API Access** (configured in .env)

### Installation & Launch

1. **Clone and Navigate**
   ```bash
   git clone <repository-url>
   cd meio-sentiment-analysis
   ```

2. **Configure Environment**
   ```bash
   # Edit .env file with your LLM API settings
   LLM_API_URL=http://60.51.17.97:9501/v1
   LLM_MODEL_NAME=llm_model
   ```

3. **Start System**
   ```bash
   ./start.sh
   ```

4. **Access Application**
   - Frontend: http://localhost:3006
   - Backend API: http://localhost:8011/docs

## Usage Guide

### Step 1: Upload Data
- Drag & drop or select CSV/XLSX files
- System supports up to 13,000 rows
- Automatic file validation and preview

### Step 2: Review Data
- Interactive table view
- Column mapping verification
- Data quality checks

### Step 3: Sentiment Analysis
- AI-powered classification
- Real-time progress tracking
- Statistical overview

### Step 4: Strategic Mitigation
- **Negative Sentiment**: Counter-narrative generation
- **Positive Sentiment**: Amplification strategies
- **Neutral Sentiment**: Positive-leaning responses

### Step 5: Intelligence Report
- Comprehensive PowerPoint generation
- Executive summary with statistics
- Detailed analysis per data point
- Downloadable PPTX format

## API Documentation

### Endpoints

#### `POST /upload`
Upload CSV/XLSX file for analysis
```json
{
  "session_id": "uuid",
  "filename": "data.csv",
  "row_count": 1000,
  "columns": ["text", "source", "date"],
  "preview": [...]
}
```

#### `POST /analyze`
Perform sentiment analysis
```json
{
  "session_id": "uuid"
}
```

#### `POST /mitigate`
Generate strategic responses
```json
{
  "session_id": "uuid"
}
```

#### `POST /generate-report`
Create PowerPoint report
- Returns: Binary PPTX file

## Configuration

### Environment Variables (.env)
```bash
# LLM API Configuration
LLM_API_URL=http://your-llm-endpoint/v1
LLM_MODEL_NAME=your_model_name

# Server Configuration
BACKEND_HOST=localhost
BACKEND_PORT=8000
FRONTEND_PORT=3000
```

### Backend Configuration
- **Concurrent Processing**: Configurable semaphore limits
- **Session Management**: In-memory storage for active sessions
- **File Processing**: Pandas-based CSV/XLSX handling
- **Report Generation**: Python-PPTX for PowerPoint creation

### Frontend Configuration
- **Styled Components**: CSS-in-JS theming
- **Axios**: HTTP client with request/response interceptors
- **React Dropzone**: Advanced file upload handling
- **Lucide Icons**: Professional icon library

## Security Considerations

### Data Handling
- **No Database Storage** - All data processed in-memory
- **Session-Based** - Temporary storage with UUID sessions
- **File Validation** - Strict file type and size limits

### API Security
- **CORS Configuration** - Restricted to frontend domain
- **Input Validation** - Pydantic models for request validation
- **Error Handling** - Secure error messages without data leakage

### Network Security
- **Local Deployment** - Designed for internal network use
- **Configurable Endpoints** - Environment-based URL configuration
- **Rate Limiting** - Built-in semaphore controls

## Performance Optimization

### Concurrent Processing
```python
semaphore = asyncio.Semaphore(10)  # Concurrent request limit
tasks = [analyze_with_semaphore(row) for row in data]
results = await asyncio.gather(*tasks)
```

### Memory Management
- **Streaming File Processing** - No full file loading
- **Session Cleanup** - Automatic garbage collection
- **Chunked Processing** - Batch processing for large datasets

### Frontend Optimization
- **Lazy Loading** - Progressive data rendering
- **Virtual Scrolling** - Efficient large table handling
- **State Management** - Optimized React state updates

## Troubleshooting

### Common Issues

**Connection Refused (Backend)**
```bash
# Check if backend is running
curl http://localhost:8011/

# Restart backend
cd backend && python main.py
```

**File Upload Errors**
- Verify file format (CSV/XLSX only)
- Check file size limits
- Ensure proper column headers

**LLM API Errors**
- Verify .env configuration
- Test API endpoint availability
- Check authentication credentials

### Logs and Monitoring
```bash
# Backend logs
cd backend && python main.py

# Frontend logs
cd frontend && npm start

# System logs
tail -f /var/log/system.log
```

## Development

### Project Structure
```
meio-sentiment-analysis/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── venv/               # Virtual environment
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   └── index.js        # React entry point
│   ├── public/
│   │   └── index.html      # HTML template
│   └── package.json        # Node.js dependencies
├── .env                    # Environment configuration
├── start.sh               # Startup script
└── README.md              # This file
```

### Adding Features

**New Analysis Types**
1. Extend LLM prompts in `call_llm_api()`
2. Add new endpoints in FastAPI
3. Update React UI components

**Additional File Formats**
1. Update file validation in backend
2. Add pandas readers for new formats
3. Update frontend dropzone configuration

## License

This project is classified as **CONFIDENTIAL** and proprietary to the Malaysian External Intelligence Organisation (MEIO). Unauthorized access, distribution, or modification is strictly prohibited.

## Support

For technical support and operational queries:
- **Internal Systems**: Contact MEIO IT Operations
- **Deployment Issues**: Refer to troubleshooting section
- **Feature Requests**: Submit through official channels

---

**Malaysian External Intelligence Organisation (MEIO)**  
*Protecting National Interests Through Advanced Intelligence Analytics*