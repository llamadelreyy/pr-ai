#!/bin/bash

# MEIO Sentiment Analysis System Startup Script
# This script starts both the FastAPI backend and React frontend

echo "🇲🇾 Malaysian External Intelligence Organisation (MEIO)"
echo "🔍 Sentiment Analysis System"
echo "=============================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Create Python virtual environment if it doesn't exist
if [ ! -d "backend/venv" ]; then
    echo "📦 Creating Python virtual environment..."
    cd backend
    python3 -m venv venv
    cd ..
fi

# Activate virtual environment and install backend dependencies
echo "📦 Installing backend dependencies..."
cd backend
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "🚀 Starting MEIO Sentiment Analysis System..."
echo ""
echo "Backend will run on: http://localhost:8011"
echo "Frontend will run on: http://localhost:3006"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend in background
echo "🔧 Starting FastAPI backend..."
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend in background
echo "🎨 Starting React frontend..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Both services are starting up..."
echo "🌐 Open http://localhost:3006 in your browser to access the application"
echo ""
echo "📊 System Status:"
echo "   - Backend PID: $BACKEND_PID"
echo "   - Frontend PID: $FRONTEND_PID"
echo ""
echo "💡 To test the backend API directly, visit: http://localhost:8011/docs"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID