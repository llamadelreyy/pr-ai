
import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  Eye,
  BarChart3,
  MessageSquare,
  FileText,
  Download,
  CheckCircle,
  AlertCircle,
  Smile,
  Frown,
  Minus,
  Loader,
  Tag,
  TrendingUp,
  Menu,
  Home,
  Database,
  Settings,
  LogOut,
  Shield,
  Users,
  Activity,
  Filter,
  Search,
  RefreshCw,
  Trash2,
  Play,
  FileSearch,
  BookOpen,
  Send,
  ArrowLeft
} from 'lucide-react';

// Styled Components
const AppContainer = styled.div`
  min-height: 100vh;
  background: #ffffff;
  color: #1f2937;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  display: flex;
`;

const Sidebar = styled.div`
  width: ${props => props.$collapsed ? '60px' : '280px'};
  background: #2596be;
  color: white;
  transition: width 0.3s ease;
  display: flex;
  flex-direction: column;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
`;

const SidebarHeader = styled.div`
  padding: 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const SidebarToggle = styled.button`
  background: none;
  border: none;
  color: white;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 4px;
  transition: background-color 0.2s;
  
  &:hover {
    background: rgba(255, 255, 255, 0.1);
  }
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 1.1rem;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
`;

const LogoIcon = styled.div`
  width: 32px;
  height: 32px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
`;

const SidebarNav = styled.nav`
  flex: 1;
  padding: 1rem 0;
`;

const NavItem = styled.div`
  padding: 0.75rem 1.5rem;
  cursor: pointer;
  transition: background-color 0.2s;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.9rem;
  background: ${props => props.$active ? 'rgba(255, 255, 255, 0.15)' : 'transparent'};
  
  &:hover {
    background: rgba(255, 255, 255, 0.1);
  }
  
  svg {
    flex-shrink: 0;
  }
  
  span {
    white-space: nowrap;
    overflow: hidden;
    opacity: ${props => props.$collapsed ? '0' : '1'};
    transition: opacity 0.3s ease;
  }
`;

const MainContent = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
`;

const Header = styled.header`
  background: white;
  padding: 1rem 2rem;
  border-bottom: 1px solid #e5e7eb;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const HeaderTitle = styled.h1`
  font-size: 1.5rem;
  font-weight: 600;
  color: #1f2937;
`;

const UserSection = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const UserAvatar = styled.div`
  width: 40px;
  height: 40px;
  background: #2596be;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 600;
`;

const LogoutButton = styled.button`
  background: #ef4444;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.875rem;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: background-color 0.2s;
  
  &:hover {
    background: #dc2626;
  }
`;

const Content = styled.main`
  flex: 1;
  padding: 2rem;
  overflow-y: auto;
  background: #f9fafb;
`;

const Card = styled.div`
  background: white;
  border-radius: 8px;
  padding: 2rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
  margin-bottom: 2rem;
`;

const LoginContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
`;

const LoginCard = styled.div`
  background: white;
  border-radius: 12px;
  padding: 3rem;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
  width: 100%;
  max-width: 400px;
`;

const LoginHeader = styled.div`
  text-align: center;
  margin-bottom: 2rem;
`;

const LoginLogo = styled.div`
  width: 80px;
  height: 80px;
  background: #2596be;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1rem;
`;

const LoginTitle = styled.h1`
  font-size: 1.5rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 0.5rem;
`;

const LoginSubtitle = styled.p`
  color: #6b7280;
  font-size: 0.875rem;
`;

const FormGroup = styled.div`
  margin-bottom: 1.5rem;
`;

const Label = styled.label`
  display: block;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
  margin-bottom: 0.5rem;
`;

const Input = styled.input`
  width: 100%;
  padding: 0.75rem 1rem;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 0.875rem;
  transition: border-color 0.2s;
  
  &:focus {
    outline: none;
    border-color: #2596be;
    box-shadow: 0 0 0 3px rgba(37, 150, 190, 0.1);
  }
`;

const Select = styled.select`
  width: 100%;
  padding: 0.75rem 1rem;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 0.875rem;
  background: white;
  transition: border-color 0.2s;
  
  &:focus {
    outline: none;
    border-color: #2596be;
    box-shadow: 0 0 0 3px rgba(37, 150, 190, 0.1);
  }
`;

const Button = styled.button`
  width: ${props => props.$fullWidth ? '100%' : 'auto'};
  background: ${props => props.$variant === 'secondary' ? 'white' : props.$variant === 'danger' ? '#ef4444' : '#2596be'};
  color: ${props => props.$variant === 'secondary' ? '#2596be' : 'white'};
  border: ${props => props.$variant === 'secondary' ? '1px solid #2596be' : 'none'};
  padding: 0.75rem 1.5rem;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  transition: all 0.2s ease;
  
  &:hover {
    background: ${props => {
      if (props.$variant === 'secondary') return '#f8fafc';
      if (props.$variant === 'danger') return '#dc2626';
      return '#1e88e5';
    }};
    transform: translateY(-1px);
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
`;

const ErrorMessage = styled.div`
  color: #dc2626;
  font-size: 0.875rem;
  margin-top: 0.5rem;
  text-align: center;
`;

const SuccessMessage = styled.div`
  color: #059669;
  font-size: 0.875rem;
  margin-top: 0.5rem;
  text-align: center;
`;

const DropZone = styled.div`
  border: 3px dashed ${props => props.$isDragActive ? '#2596be' : '#d1d5db'};
  border-radius: 8px;
  padding: 3rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: ${props => props.$isDragActive ? '#f0f9ff' : '#fafbfc'};
  
  &:hover {
    border-color: #2596be;
    background: #f0f9ff;
  }
`;

const TableContainer = styled.div`
  background: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
  margin: 2rem auto;
  max-width: 1200px;
`;

const TableHeader = styled.div`
  background: #f8fafc;
  padding: 1rem 2rem;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
`;

const TableTitle = styled.h3`
  font-size: 1.1rem;
  font-weight: 600;
  color: #1f2937;
`;

const TableActions = styled.div`
  display: flex;
  gap: 0.5rem;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
`;

const Th = styled.th`
  background: #f8fafc;
  padding: 1rem;
  text-align: left;
  font-weight: 600;
  color: #374151;
  border-bottom: 1px solid #e5e7eb;
  font-size: 0.875rem;
`;

const Td = styled.td`
  padding: 1rem;
  border-bottom: 1px solid #e5e7eb;
  vertical-align: top;
  font-size: 0.875rem;
  color: #374151;
  max-width: 400px;
  word-wrap: break-word;
`;

const ContentCell = styled(Td)`
  max-width: 500px;
  line-height: 1.5;
`;

const SentimentBadge = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  background: ${props => {
    switch(props.$sentiment) {
      case 'positive': return '#dcfce7';
      case 'negative': return '#fee2e2';
      case 'neutral': return '#f3f4f6';
      default: return '#f3f4f6';
    }
  }};
  color: ${props => {
    switch(props.$sentiment) {
      case 'positive': return '#166534';
      case 'negative': return '#991b1b';
      case 'neutral': return '#374151';
      default: return '#374151';
    }
  }};
`;

const TopicBadge = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 500;
  background: #eff6ff;
  color: #1e40af;
  border: 1px solid #bfdbfe;
  margin: 0.125rem;
`;

const TagsContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
`;

const StatusBadge = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  background: ${props => {
    switch(props.$status) {
      case 'uploaded': return '#fef3c7';
      case 'analyzed': return '#dbeafe';
      case 'completed': return '#dcfce7';
      default: return '#f3f4f6';
    }
  }};
  color: ${props => {
    switch(props.$status) {
      case 'uploaded': return '#92400e';
      case 'analyzed': return '#1e40af';
      case 'completed': return '#166534';
      default: return '#374151';
    }
  }};
`;

const LoadingSpinner = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin: 2rem 0;
  color: #6b7280;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 2rem;
  flex-wrap: wrap;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const StatCard = styled.div`
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  text-align: center;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
`;

const StatNumber = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: ${props => props.color || '#2596be'};
  margin-bottom: 0.5rem;
`;

const StatLabel = styled.div`
  font-size: 0.875rem;
  color: #6b7280;
  font-weight: 500;
`;

const DocumentCard = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  transition: box-shadow 0.2s;
  
  &:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
`;

const DocumentHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
`;

const DocumentTitle = styled.h3`
  font-size: 1.1rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 0.5rem;
`;

const DocumentMeta = styled.div`
  color: #6b7280;
  font-size: 0.875rem;
`;

const StreamingContainer = styled.div`
  background: #f8fafc;
  border-radius: 8px;
  padding: 1.5rem;
  margin: 1rem 0;
  border: 1px solid #e5e7eb;
  max-height: 400px;
  overflow-y: auto;
`;

const StreamingRow = styled.div`
  padding: 1rem;
  border-bottom: 1px solid #e5e7eb;
  margin-bottom: 1rem;
  border-radius: 6px;
  background: white;
`;

const StreamingText = styled.div`
  color: #10b981;
  font-size: 0.875rem;
  line-height: 1.5;
  
  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }
`;

// Debug: Log the API URL being used
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8011';
console.log('Frontend is using API_BASE:', API_BASE);
console.log('Environment variable REACT_APP_API_URL:', process.env.REACT_APP_API_URL);

// Alternative: Try to detect if we're in a port-forwarded environment
const isPortForwarded = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
const FALLBACK_API_BASE = isPortForwarded
  ? `http://${window.location.hostname}:8011`
  : 'http://localhost:8011';

const FINAL_API_BASE = process.env.REACT_APP_API_URL || FALLBACK_API_BASE;
console.log('Final API_BASE being used:', FINAL_API_BASE);

function App() {
  // Authentication state
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loginForm, setLoginForm] = useState({ username: '', password: '' });
  const [loginError, setLoginError] = useState('');

  // UI state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeNav, setActiveNav] = useState('dashboard');
  
  // Application state
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState('');
  const [currentDocumentData, setCurrentDocumentData] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [responses, setResponses] = useState(null);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState(''); // 'success' or 'error'
  
  // Streaming states
  const [analysisStreamingMode, setAnalysisStreamingMode] = useState(false);
  const [analysisStreamingData, setAnalysisStreamingData] = useState([]);
  const [analysisStreamingStatus, setAnalysisStreamingStatus] = useState("");
  const [responseStreamingMode, setResponseStreamingMode] = useState(false);
  const [responseStreamingData, setResponseStreamingData] = useState([]);
  const [responseStreamingStatus, setResponseStreamingStatus] = useState("");

  // Check for saved authentication on mount
  useEffect(() => {
    const savedAuth = localStorage.getItem('meioAuth');
    if (savedAuth === 'true') {
      setIsAuthenticated(true);
      loadDocuments();
    }
  }, []);

  const showMessage = (text, type = 'success') => {
    setMessage(text);
    setMessageType(type);
    setTimeout(() => {
      setMessage('');
      setMessageType('');
    }, 3000);
  };

  const loadDocuments = async () => {
    try {
      // Load from localStorage since backend uses sessions
      const existingSessions = JSON.parse(localStorage.getItem('meioSessions') || '[]');
      setDocuments(existingSessions);
    } catch (error) {
      console.error('Error loading documents:', error);
    }
  };

  const handleLogin = (e) => {
    e.preventDefault();
    setLoginError('');
    
    if (loginForm.username === 'admin' && loginForm.password === 'meio2024') {
      setIsAuthenticated(true);
      localStorage.setItem('meioAuth', 'true');
      loadDocuments();
    } else {
      setLoginError('Invalid credentials. Use admin/meio2024');
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    localStorage.removeItem('meioAuth');
    setActiveNav('dashboard');
    setDocuments([]);
    setSelectedDocument('');
    setCurrentDocumentData(null);
    setAnalysisResults(null);
    setResponses(null);
  };

  const onDrop = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${FINAL_API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      showMessage(`Document "${file.name}" uploaded successfully!`);
      // Store the session data with session_id from backend
      const sessionData = {
        id: response.data.session_id,
        session_id: response.data.session_id, // Keep session_id for backend calls
        filename: response.data.filename,
        row_count: response.data.row_count,
        upload_date: new Date().toISOString(),
        status: 'uploaded'
      };
      
      const existingSessions = JSON.parse(localStorage.getItem('meioSessions') || '[]');
      existingSessions.push(sessionData);
      localStorage.setItem('meioSessions', JSON.stringify(existingSessions));
      
      loadDocuments();
    } catch (error) {
      showMessage('Error uploading file: ' + (error.response?.data?.detail || error.message), 'error');
    }
    setLoading(false);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    },
    multiple: false
  });

  const viewDocument = async (documentId) => {
    setLoading(true);
    try {
      const response = await axios.get(`${FINAL_API_BASE}/data/${documentId}`);
      setCurrentDocumentData({
        id: documentId,
        ...response.data,
        columns: Object.keys(response.data.data[0] || {}),
        row_count: response.data.data.length
      });
      setActiveNav('data-viewer');
    } catch (error) {
      showMessage('Error loading document: ' + (error.response?.data?.detail || error.message), 'error');
    }
    setLoading(false);
  };

  const deleteDocument = async (documentId) => {
    if (!window.confirm('Are you sure you want to delete this document?')) return;
    
    setLoading(true);
    try {
      // Remove from localStorage since backend uses sessions
      const existingSessions = JSON.parse(localStorage.getItem('meioSessions') || '[]');
      const updatedSessions = existingSessions.filter(doc => doc.id !== documentId);
      localStorage.setItem('meioSessions', JSON.stringify(updatedSessions));
      
      showMessage('Document deleted successfully!');
      loadDocuments();
      if (currentDocumentData?.id === documentId) {
        setCurrentDocumentData(null);
      }
    } catch (error) {
      showMessage('Error deleting document: ' + error.message, 'error');
    }
    setLoading(false);
  };

  const analyzeDocument = async () => {
    if (!selectedDocument) {
      showMessage('Please select a document to analyze', 'error');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${FINAL_API_BASE}/analyze`, {
        session_id: selectedDocument
      });
      setAnalysisResults(response.data);
      showMessage('Document analyzed successfully!');
      
      // Update status in localStorage
      const existingSessions = JSON.parse(localStorage.getItem('meioSessions') || '[]');
      const updatedSessions = existingSessions.map(doc =>
        doc.id === selectedDocument ? {...doc, status: 'analyzed'} : doc
      );
      localStorage.setItem('meioSessions', JSON.stringify(updatedSessions));
      loadDocuments();
    } catch (error) {
      showMessage('Error analyzing document: ' + (error.response?.data?.detail || error.message), 'error');
    }
    setLoading(false);
  };

  const analyzeDocumentStreaming = async () => {
    if (!selectedDocument) {
      showMessage('Please select a document to analyze', 'error');
      return;
    }

    setAnalysisStreamingMode(true);
    setAnalysisStreamingData([]);
    setAnalysisStreamingStatus("Starting sentiment analysis...");
    setLoading(true);
    
    try {
      const response = await fetch(`${FINAL_API_BASE}/analyze-streaming`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: selectedDocument
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              switch(data.type) {
                case 'row_start':
                  setAnalysisStreamingStatus(`Analyzing row ${data.index + 1}/${data.total}: ${data.text}...`);
                  setAnalysisStreamingData(prev => [...prev, {
                    index: data.index,
                    text: data.text,
                    sentiment: null,
                    confidence: null,
                    topics: [],
                    complete: false
                  }]);
                  break;
                  
                case 'sentiment':
                  setAnalysisStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, sentiment: data.sentiment, confidence: data.confidence}
                      : item
                  ));
                  break;
                  
                case 'topic':
                  setAnalysisStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, topics: data.topic.split(',').map(t => t.trim())}
                      : item
                  ));
                  break;
                  
                case 'row_complete':
                  setAnalysisStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, sentiment: data.sentiment, confidence: data.confidence, topics: data.topic.split(',').map(t => t.trim()), complete: true}
                      : item
                  ));
                  break;
                  
                case 'error':
                  setAnalysisStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, sentiment: 'error', confidence: 0, topics: ['Error'], complete: true}
                      : item
                  ));
                  break;
                  
                case 'complete_all':
                  setAnalysisResults(data);
                  setAnalysisStreamingStatus("Sentiment analysis completed!");
                  showMessage('Document analyzed successfully with streaming!');
                  loadDocuments(); // Refresh to update status
                  setLoading(false);
                  setTimeout(() => {
                    setAnalysisStreamingMode(false);
                  }, 3000);
                  return;
              }
            } catch (error) {
              console.error('Error parsing streaming data:', error);
            }
          }
        }
      }
      
    } catch (error) {
      console.error('Streaming error:', error);
      setAnalysisStreamingStatus("Error occurred during streaming: " + error.message);
      showMessage('Error during streaming analysis: ' + error.message, 'error');
      setLoading(false);
      setAnalysisStreamingMode(false);
    }
  };

  const generateResponses = async () => {
    if (!selectedDocument) {
      showMessage('Please select a document to generate responses for', 'error');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${FINAL_API_BASE}/mitigate`, {
        session_id: selectedDocument
      });
      
      // Format the response data to match expected structure
      const formattedResponses = {
        responses: response.data.data_with_comments.map((item, idx) => ({
          original_text: Object.values(item).find(val =>
            typeof val === 'string' && val.length > 10 &&
            !['sentiment', 'confidence', 'topic', 'generated_comment'].includes(val)
          ) || 'No text content',
          sentiment: item.sentiment,
          topics: Array.isArray(item.topics) ? item.topics : [item.topic || 'General'],
          generated_comment: item.generated_comment
        }))
      };
      
      setResponses(formattedResponses);
      showMessage('Responses generated successfully!');
      
      // Update status in localStorage
      const existingSessions = JSON.parse(localStorage.getItem('meioSessions') || '[]');
      const updatedSessions = existingSessions.map(doc =>
        doc.id === selectedDocument ? {...doc, status: 'completed'} : doc
      );
      localStorage.setItem('meioSessions', JSON.stringify(updatedSessions));
      loadDocuments();
    } catch (error) {
      showMessage('Error generating responses: ' + (error.response?.data?.detail || error.message), 'error');
    }
    setLoading(false);
  };

  const generateResponsesStreaming = async () => {
    if (!selectedDocument) {
      showMessage('Please select a document to generate responses for', 'error');
      return;
    }

    setResponseStreamingMode(true);
    setResponseStreamingData([]);
    setResponseStreamingStatus("Starting response generation...");
    setLoading(true);
    
    try {
      const response = await fetch(`${FINAL_API_BASE}/mitigate-streaming`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: selectedDocument
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              switch(data.type) {
                case 'row_start':
                  setResponseStreamingStatus(`Processing row ${data.index + 1}/${data.total}: ${data.text}...`);
                  setResponseStreamingData(prev => [...prev, {
                    index: data.index,
                    text: data.text,
                    sentiment: data.sentiment,
                    topic: data.topic,
                    comment: "",
                    complete: false
                  }]);
                  break;
                  
                case 'word':
                  setResponseStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, comment: data.complete}
                      : item
                  ));
                  break;
                  
                case 'row_complete':
                  setResponseStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, comment: data.comment, complete: true}
                      : item
                  ));
                  break;
                  
                case 'error':
                  setResponseStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, comment: `Error: ${data.error}`, complete: true}
                      : item
                  ));
                  break;
                  
                case 'complete_all':
                  // Extract text content from each item for proper display
                  const formattedData = data.data.map((item, idx) => {
                    // Find the main text content (not generated_comment, sentiment, etc.)
                    let originalText = '';
                    for (const [key, value] of Object.entries(item)) {
                      if (key !== 'generated_comment' && key !== 'sentiment' && key !== 'confidence' && key !== 'topic' &&
                          typeof value === 'string' && value.length > 10) {
                        originalText = value;
                        break;
                      }
                    }
                    
                    return {
                      original_text: originalText || 'No text content',
                      sentiment: item.sentiment || 'neutral',
                      topics: Array.isArray(item.topics) ? item.topics : (item.topic ? item.topic.split(',').map(t => t.trim()) : ['General']),
                      generated_comment: item.generated_comment || 'No comment generated'
                    };
                  });
                  
                  setResponses({
                    responses: formattedData
                  });
                  setResponseStreamingStatus("Response generation completed!");
                  showMessage('Responses generated successfully with streaming!');
                  // Update status in localStorage
                  const existingSessions = JSON.parse(localStorage.getItem('meioSessions') || '[]');
                  const updatedSessions = existingSessions.map(doc =>
                    doc.id === selectedDocument ? {...doc, status: 'completed'} : doc
                  );
                  localStorage.setItem('meioSessions', JSON.stringify(updatedSessions));
                  loadDocuments();
                  setLoading(false);
                  setTimeout(() => {
                    setResponseStreamingMode(false);
                  }, 3000);
                  return;
              }
            } catch (error) {
              console.error('Error parsing streaming data:', error);
            }
          }
        }
      }
      
    } catch (error) {
      console.error('Streaming error:', error);
      setResponseStreamingStatus("Error occurred during streaming: " + error.message);
      showMessage('Error during streaming response generation: ' + error.message, 'error');
      setLoading(false);
      setResponseStreamingMode(false);
    }
  };

  const generateReport = async () => {
    if (!selectedDocument) {
      showMessage('Please select a document to generate report for', 'error');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${FINAL_API_BASE}/generate-report`, {
        session_id: selectedDocument
      }, {
        responseType: 'blob'
      });
      
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Extract filename from response headers or use default
      const contentDisposition = response.headers['content-disposition'];
      let filename = `MEIO_Report_${new Date().toISOString().slice(0,10)}.pptx`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
      
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      showMessage('Report generated and downloaded successfully!');
    } catch (error) {
      showMessage('Error generating report: ' + (error.response?.data?.detail || error.message), 'error');
    }
    setLoading(false);
  };

  const getPageTitle = () => {
    switch (activeNav) {
      case 'dashboard': return 'Dashboard Overview';
      case 'data-upload': return 'Data Upload';
      case 'data-viewer': return 'Data Viewer';
      case 'data-analysis': return 'Data Analysis';
      case 'response-generation': return 'Response Generation';
      case 'report-generation': return 'Report Generation';
      case 'user-management': return 'User Management';
      default: return 'MEIO Sentiment Analysis System';
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch(sentiment) {
      case 'positive': return <Smile size={14} />;
      case 'negative': return <Frown size={14} />;
      case 'neutral': return <Minus size={14} />;
      default: return <Minus size={14} />;
    }
  };

  // Login Screen
  if (!isAuthenticated) {
    return (
      <LoginContainer>
        <LoginCard>
          <LoginHeader>
            <LoginLogo>
              <Shield size={40} color="white" />
            </LoginLogo>
            <LoginTitle>MEIO Portal</LoginTitle>
            <LoginSubtitle>Malaysian External Intelligence Organisation</LoginSubtitle>
          </LoginHeader>
          
          <form onSubmit={handleLogin}>
            <FormGroup>
              <Label>Username</Label>
              <Input
                type="text"
                value={loginForm.username}
                onChange={(e) => setLoginForm({...loginForm, username: e.target.value})}
                placeholder="Enter your username"
                required
              />
            </FormGroup>
            
            <FormGroup>
              <Label>Password</Label>
              <Input
                type="password"
                value={loginForm.password}
                onChange={(e) => setLoginForm({...loginForm, password: e.target.value})}
                placeholder="Enter your password"
                required
              />
            </FormGroup>
            
            {loginError && <ErrorMessage>{loginError}</ErrorMessage>}
            
            <Button type="submit" $fullWidth>
              <Shield size={16} />
              Sign In
            </Button>
            
            <div style={{marginTop: '1rem', fontSize: '0.75rem', color: '#6b7280', textAlign: 'center'}}>
              Demo credentials: admin / meio2024
            </div>
          </form>
        </LoginCard>
      </LoginContainer>
    );
  }

  // Main Application
  return (
    <AppContainer>
      <Sidebar $collapsed={sidebarCollapsed}>
        <SidebarHeader>
          <SidebarToggle onClick={() => setSidebarCollapsed(!sidebarCollapsed)}>
            <Menu size={20} />
          </SidebarToggle>
          {!sidebarCollapsed && (
            <Logo>
              <LogoIcon>
                <Shield size={20} />
              </LogoIcon>
              <span>MEIO Portal</span>
            </Logo>
          )}
        </SidebarHeader>
        
        <SidebarNav>
          <NavItem 
            $active={activeNav === 'dashboard'} 
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('dashboard')}
          >
            <Home size={20} />
            <span>Dashboard</span>
          </NavItem>
          
          <NavItem 
            $active={activeNav === 'data-upload'} 
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('data-upload')}
          >
            <Upload size={20} />
            <span>Data Upload</span>
          </NavItem>
          
          <NavItem 
            $active={activeNav === 'data-viewer'} 
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('data-viewer')}
          >
            <FileSearch size={20} />
            <span>Data Viewer</span>
          </NavItem>
          
          <NavItem 
            $active={activeNav === 'data-analysis'} 
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('data-analysis')}
          >
            <BarChart3 size={20} />
            <span>Data Analysis</span>
          </NavItem>
          
          <NavItem 
            $active={activeNav === 'response-generation'} 
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('response-generation')}
          >
            <Send size={20} />
            <span>Response Generation</span>
          </NavItem>
          
          <NavItem 
            $active={activeNav === 'report-generation'} 
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('report-generation')}
          >
            <BookOpen size={20} />
            <span>Report Generation</span>
          </NavItem>
          
          <NavItem 
            $active={activeNav === 'user-management'} 
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('user-management')}
          >
            <Users size={20} />
            <span>User Management</span>
          </NavItem>
        </SidebarNav>
      </Sidebar>
      
      <MainContent>
        <Header>
          <HeaderTitle>{getPageTitle()}</HeaderTitle>
          <UserSection>
            <UserAvatar>A</UserAvatar>
            <span style={{fontSize: '0.875rem', color: '#6b7280'}}>Administrator</span>
            <LogoutButton onClick={handleLogout}>
              <LogOut size={16} />
              Logout
            </LogoutButton>
          </UserSection>
        </Header>
        
        <Content>
          {message && (
            <Card style={{marginBottom: '1rem', background: messageType === 'error' ? '#fee2e2' : '#dcfce7', border: messageType === 'error' ? '1px solid #fca5a5' : '1px solid #86efac'}}>
              <div style={{color: messageType === 'error' ? '#991b1b' : '#166534', fontWeight: '500'}}>
                {message}
              </div>
            </Card>
          )}

          {loading && (
            <LoadingSpinner>
              <Loader className="animate-spin" size={24} />
              Processing...
            </LoadingSpinner>
          )}

          {/* Dashboard */}
          {activeNav === 'dashboard' && (
            <div>
              <StatsGrid>
                <StatCard>
                  <StatNumber color="#2596be">{documents.length}</StatNumber>
                  <StatLabel>Total Documents</StatLabel>
                </StatCard>
                <StatCard>
                  <StatNumber color="#10b981">{documents.filter(d => d.status === 'completed').length}</StatNumber>
                  <StatLabel>Completed Analysis</StatLabel>
                </StatCard>
                <StatCard>
                  <StatNumber color="#f59e0b">{documents.filter(d => d.status === 'analyzed').length}</StatNumber>
                  <StatLabel>Awaiting Response</StatLabel>
                </StatCard>
                <StatCard>
                  <StatNumber color="#ef4444">{documents.filter(d => d.status === 'uploaded').length}</StatNumber>
                  <StatLabel>Pending Analysis</StatLabel>
                </StatCard>
              </StatsGrid>

              <Card>
                <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Recent Documents</h2>
                {documents.length === 0 ? (
                  <p style={{color: '#6b7280', textAlign: 'center', padding: '2rem'}}>
                    No documents uploaded yet. Start by uploading a document.
                  </p>
                ) : (
                  documents.slice(0, 5).map(doc => (
                    <DocumentCard key={doc.id}>
                      <DocumentHeader>
                        <div>
                          <DocumentTitle>{doc.filename}</DocumentTitle>
                          <DocumentMeta>
                            Uploaded: {new Date(doc.upload_date).toLocaleDateString()} |
                            Rows: {doc.row_count} |
                            Status: <StatusBadge $status={doc.status}>{doc.status}</StatusBadge>
                          </DocumentMeta>
                        </div>
                        <div style={{display: 'flex', gap: '0.5rem'}}>
                          <Button $variant="secondary" onClick={() => viewDocument(doc.id)}>
                            <Eye size={16} />
                            View
                          </Button>
                        </div>
                      </DocumentHeader>
                    </DocumentCard>
                  ))
                )}
              </Card>
            </div>
          )}

          {/* Data Upload */}
          {activeNav === 'data-upload' && (
            <Card>
              <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Upload New Document</h2>
              <DropZone {...getRootProps()} $isDragActive={isDragActive}>
                <input {...getInputProps()} />
                <Upload size={48} style={{margin: '0 auto 1rem', color: '#2596be'}} />
                <h3 style={{color: '#1f2937', marginBottom: '0.5rem'}}>Drop your CSV or XLSX file here</h3>
                <p style={{color: '#6b7280', fontSize: '0.875rem'}}>
                  Or click to select file (Max 13,000 rows supported)
                </p>
              </DropZone>

              <div style={{marginTop: '2rem'}}>
                <h3 style={{marginBottom: '1rem', color: '#1f2937'}}>Upload History</h3>
                {documents.length === 0 ? (
                  <p style={{color: '#6b7280', textAlign: 'center', padding: '2rem'}}>
                    No documents uploaded yet.
                  </p>
                ) : (
                  documents.map(doc => (
                    <DocumentCard key={doc.id}>
                      <DocumentHeader>
                        <div>
                          <DocumentTitle>{doc.filename}</DocumentTitle>
                          <DocumentMeta>
                            Uploaded: {new Date(doc.upload_date).toLocaleDateString()} |
                            Rows: {doc.row_count} |
                            Status: <StatusBadge $status={doc.status}>{doc.status}</StatusBadge>
                          </DocumentMeta>
                        </div>
                        <div style={{display: 'flex', gap: '0.5rem'}}>
                          <Button $variant="secondary" onClick={() => viewDocument(doc.id)}>
                            <Eye size={16} />
                            View
                          </Button>
                          <Button $variant="danger" onClick={() => deleteDocument(doc.id)}>
                            <Trash2 size={16} />
                            Delete
                          </Button>
                        </div>
                      </DocumentHeader>
                    </DocumentCard>
                  ))
                )}
              </div>
            </Card>
          )}

          {/* Data Viewer */}
          {activeNav === 'data-viewer' && (
            <div>
              {!currentDocumentData ? (
                <Card>
                  <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Select Document to View</h2>
                  {documents.length === 0 ? (
                    <p style={{color: '#6b7280', textAlign: 'center', padding: '2rem'}}>
                      No documents available. Upload a document first.
                    </p>
                  ) : (
                    documents.map(doc => (
                      <DocumentCard key={doc.id}>
                        <DocumentHeader>
                          <div>
                            <DocumentTitle>{doc.filename}</DocumentTitle>
                            <DocumentMeta>
                              Uploaded: {new Date(doc.upload_date).toLocaleDateString()} |
                              Rows: {doc.row_count} |
                              Status: <StatusBadge $status={doc.status}>{doc.status}</StatusBadge>
                            </DocumentMeta>
                          </div>
                          <Button onClick={() => viewDocument(doc.id)}>
                            <Eye size={16} />
                            View Data
                          </Button>
                        </DocumentHeader>
                      </DocumentCard>
                    ))
                  )}
                </Card>
              ) : (
                <div>
                  <Card>
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem'}}>
                      <h2 style={{color: '#1f2937', margin: 0}}>{currentDocumentData.filename}</h2>
                      <Button $variant="secondary" onClick={() => setCurrentDocumentData(null)}>
                        <ArrowLeft size={16} />
                        Back to List
                      </Button>
                    </div>
                    
                    <DocumentMeta style={{marginBottom: '1.5rem'}}>
                      Uploaded: {new Date(currentDocumentData.upload_date).toLocaleDateString()} |
                      Rows: {currentDocumentData.row_count} |
                      Columns: {currentDocumentData.columns?.length}
                    </DocumentMeta>
                  </Card>

                  <TableContainer>
                    <TableHeader>
                      <TableTitle>Document Data ({currentDocumentData.row_count} rows)</TableTitle>
                      <TableActions>
                        <Button $variant="secondary" style={{padding: '0.5rem'}}>
                          <Filter size={16} />
                        </Button>
                        <Button $variant="secondary" style={{padding: '0.5rem'}}>
                          <Search size={16} />
                        </Button>
                      </TableActions>
                    </TableHeader>
                    <Table>
                      <thead>
                        <tr>
                          {currentDocumentData.columns?.map(col => (
                            <Th key={col}>{col}</Th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {currentDocumentData.data?.slice(0, 50).map((row, idx) => (
                          <tr key={idx}>
                            {currentDocumentData.columns?.map(col => (
                              <Td key={col} title={row[col]}>
                                {String(row[col] || '').slice(0, 100)}
                                {String(row[col] || '').length > 100 ? '...' : ''}
                              </Td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </TableContainer>
                </div>
              )}
            </div>
          )}

          {/* Data Analysis */}
          {activeNav === 'data-analysis' && (
            <Card>
              <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Sentiment Analysis</h2>
              
              <FormGroup>
                <Label>Select Document to Analyze</Label>
                <Select
                  value={selectedDocument}
                  onChange={(e) => setSelectedDocument(e.target.value)}
                >
                  <option value="">Choose a document...</option>
                  {documents.map(doc => (
                    <option key={doc.id} value={doc.id}>
                      {doc.filename} ({doc.row_count} rows)
                    </option>
                  ))}
                </Select>
              </FormGroup>

              <ButtonGroup>
                <Button onClick={analyzeDocument} disabled={!selectedDocument || loading}>
                  <Play size={16} />
                  Start Analysis
                </Button>
                <Button onClick={analyzeDocumentStreaming} disabled={!selectedDocument || loading} style={{background: '#10b981'}}>
                  <TrendingUp size={16} />
                  Start Streaming Analysis
                </Button>
                {selectedDocument && (
                  <Button $variant="secondary" onClick={() => {
                    const doc = documents.find(d => d.id === selectedDocument);
                    if (doc) viewDocument(doc.id);
                  }}>
                    <Eye size={16} />
                    View Document
                  </Button>
                )}
              </ButtonGroup>

              {/* Analysis Streaming Interface */}
              {analysisStreamingMode && (
                <div style={{marginTop: '2rem'}}>
                  <h3 style={{marginBottom: '1rem', color: '#1f2937'}}>Real-time Sentiment Analysis</h3>
                  <div style={{marginBottom: '1rem', fontWeight: 'bold', color: '#10b981'}}>
                    {analysisStreamingStatus}
                  </div>
                  
                  <StreamingContainer>
                    {analysisStreamingData.map((item, index) => (
                      <StreamingRow key={index}>
                        <div style={{marginBottom: '0.5rem', fontWeight: '600', color: '#1f2937'}}>
                          Row {item.index + 1}:
                        </div>
                        <div style={{marginBottom: '0.5rem', fontSize: '0.875rem', color: '#6b7280'}}>
                          {item.text}
                        </div>
                        <div style={{display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap'}}>
                          {item.sentiment && (
                            <SentimentBadge $sentiment={item.sentiment}>
                              {getSentimentIcon(item.sentiment)}
                              {item.sentiment}
                            </SentimentBadge>
                          )}
                          {item.confidence !== null && (
                            <span style={{color: '#2596be', fontSize: '0.875rem'}}>
                              Confidence: {(item.confidence * 100).toFixed(1)}%
                            </span>
                          )}
                          {item.topics && item.topics.length > 0 && (
                            <TagsContainer>
                              {item.topics && item.topics.map((topic, idx) => (
                                <TopicBadge key={`topic-${item.index}-${idx}`}>
                                  <Tag size={12} />
                                  {topic}
                                </TopicBadge>
                              ))}
                            </TagsContainer>
                          )}
                          {!item.complete && (
                            <StreamingText style={{animation: 'blink 1s infinite', color: '#10b981'}}>
                              Analyzing...
                            </StreamingText>
                          )}
                        </div>
                      </StreamingRow>
                    ))}
                  </StreamingContainer>
                </div>
              )}

              {analysisResults && (
                <div style={{marginTop: '2rem'}}>
                  <h3 style={{marginBottom: '1rem', color: '#1f2937'}}>Analysis Results</h3>
                  
                  <StatsGrid>
                    <StatCard>
                      <StatNumber color="#10b981">{analysisResults.statistics?.positive || 0}</StatNumber>
                      <StatLabel>Positive ({analysisResults.analyzed_data?.length > 0 ? (((analysisResults.statistics?.positive || 0) / analysisResults.analyzed_data.length) * 100).toFixed(1) : '0.0'}%)</StatLabel>
                    </StatCard>
                    <StatCard>
                      <StatNumber color="#ef4444">{analysisResults.statistics?.negative || 0}</StatNumber>
                      <StatLabel>Negative ({analysisResults.analyzed_data?.length > 0 ? (((analysisResults.statistics?.negative || 0) / analysisResults.analyzed_data.length) * 100).toFixed(1) : '0.0'}%)</StatLabel>
                    </StatCard>
                    <StatCard>
                      <StatNumber color="#6b7280">{analysisResults.statistics?.neutral || 0}</StatNumber>
                      <StatLabel>Neutral ({analysisResults.analyzed_data?.length > 0 ? (((analysisResults.statistics?.neutral || 0) / analysisResults.analyzed_data.length) * 100).toFixed(1) : '0.0'}%)</StatLabel>
                    </StatCard>
                    <StatCard>
                      <StatNumber>{analysisResults.average_confidence && !isNaN(analysisResults.average_confidence) ? (analysisResults.average_confidence * 100).toFixed(1) : '0.0'}%</StatNumber>
                      <StatLabel>Average Confidence</StatLabel>
                    </StatCard>
                  </StatsGrid>

                  <TableContainer>
                    <TableHeader>
                      <TableTitle>Analysis Results</TableTitle>
                    </TableHeader>
                    <Table>
                      <thead>
                        <tr>
                          <Th>Content</Th>
                          <Th>Sentiment</Th>
                          <Th>Confidence</Th>
                          <Th>Topics</Th>
                        </tr>
                      </thead>
                      <tbody>
                        {analysisResults.analyzed_data?.slice(0, 50).map((row, idx) => {
                          const textContent = Object.values(row).find(val =>
                            typeof val === 'string' && val.length > 10 &&
                            !['sentiment', 'confidence', 'topic', 'topics'].includes(val)
                          ) || 'No text content';

                          const topics = Array.isArray(row.topics) ? row.topics : [row.topic || 'General'];
                          
                          return (
                            <tr key={idx}>
                              <ContentCell>{String(textContent)}</ContentCell>
                              <Td>
                                <SentimentBadge $sentiment={row.sentiment}>
                                  {getSentimentIcon(row.sentiment)}
                                  {row.sentiment}
                                </SentimentBadge>
                              </Td>
                              <Td>{row.confidence && !isNaN(row.confidence) ? (row.confidence * 100).toFixed(1) : '0.0'}%</Td>
                              <Td>
                                <TagsContainer>
                                  {topics.slice(0, 3).map((topic, topicIdx) => (
                                    <TopicBadge key={`analysis-topic-${idx}-${topicIdx}`}>
                                      <Tag size={12} />
                                      {topic}
                                    </TopicBadge>
                                  ))}
                                </TagsContainer>
                              </Td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </Table>
                  </TableContainer>
                </div>
              )}
            </Card>
          )}

          {/* Response Generation */}
          {activeNav === 'response-generation' && (
            <Card>
              <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Response Generation</h2>
              
              <FormGroup>
                <Label>Select Analyzed Document</Label>
                <Select
                  value={selectedDocument}
                  onChange={(e) => setSelectedDocument(e.target.value)}
                >
                  <option value="">Choose a document...</option>
                  {documents.map(doc => (
                    <option key={doc.id} value={doc.id}>
                      {doc.filename} ({doc.row_count} rows)
                    </option>
                  ))}
                </Select>
              </FormGroup>

              <ButtonGroup>
                <Button onClick={generateResponses} disabled={!selectedDocument || loading}>
                  <Send size={16} />
                  Generate Responses
                </Button>
                <Button onClick={generateResponsesStreaming} disabled={!selectedDocument || loading} style={{background: '#10b981'}}>
                  <TrendingUp size={16} />
                  Generate Streaming Responses
                </Button>
                {selectedDocument && (
                  <Button $variant="secondary" onClick={() => {
                    const doc = documents.find(d => d.id === selectedDocument);
                    if (doc) viewDocument(doc.id);
                  }}>
                    <Eye size={16} />
                    View Document
                  </Button>
                )}
              </ButtonGroup>

              {/* Response Streaming Interface */}
              {responseStreamingMode && (
                <div style={{marginTop: '2rem'}}>
                  <h3 style={{marginBottom: '1rem', color: '#1f2937'}}>Real-time Response Generation</h3>
                  <div style={{marginBottom: '1rem', fontWeight: 'bold', color: '#10b981'}}>
                    {responseStreamingStatus}
                  </div>
                  
                  <StreamingContainer>
                    {responseStreamingData.map((item, index) => (
                      <StreamingRow key={`response-streaming-${item.index}`}>
                        <div style={{marginBottom: '0.5rem', fontWeight: '600', color: '#1f2937'}}>
                          Row {item.index + 1}:
                          <SentimentBadge $sentiment={item.sentiment} style={{marginLeft: '0.5rem'}}>
                            {getSentimentIcon(item.sentiment)}
                            {item.sentiment}
                          </SentimentBadge>
                          <TopicBadge style={{marginLeft: '0.5rem'}}>
                            <Tag size={12} />
                            {item.topic}
                          </TopicBadge>
                        </div>
                        <div style={{marginBottom: '0.5rem', fontSize: '0.875rem', color: '#6b7280'}}>
                          {item.text}
                        </div>
                        <div style={{color: '#10b981', fontSize: '0.875rem', lineHeight: '1.5'}}>
                          {item.comment}
                          {!item.complete && <span style={{animation: 'blink 1s infinite'}}>|</span>}
                        </div>
                      </StreamingRow>
                    ))}
                  </StreamingContainer>
                </div>
              )}

              {responses && (
                <div style={{marginTop: '2rem'}}>
                  <h3 style={{marginBottom: '1rem', color: '#1f2937'}}>Generated Responses</h3>
                  
                  <TableContainer>
                    <TableHeader>
                      <TableTitle>Response Strategy</TableTitle>
                    </TableHeader>
                    <Table>
                      <thead>
                        <tr>
                          <Th>Original Content</Th>
                          <Th>Sentiment</Th>
                          <Th>Topics</Th>
                          <Th>Generated Response</Th>
                        </tr>
                      </thead>
                      <tbody>
                        {responses.responses?.slice(0, 50).map((response, idx) => (
                          <tr key={idx}>
                            <ContentCell>{response.original_text}</ContentCell>
                            <Td>
                              <SentimentBadge $sentiment={response.sentiment}>
                                {getSentimentIcon(response.sentiment)}
                                {response.sentiment}
                              </SentimentBadge>
                            </Td>
                            <Td>
                              <TagsContainer>
                                {response.topics?.slice(0, 3).map((topic, topicIdx) => (
                                  <TopicBadge key={`response-topic-${idx}-${topicIdx}`}>
                                    <Tag size={12} />
                                    {topic}
                                  </TopicBadge>
                                ))}
                              </TagsContainer>
                            </Td>
                            <ContentCell>{response.generated_comment}</ContentCell>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </TableContainer>
                </div>
              )}
            </Card>
          )}

          {/* Report Generation */}
          {activeNav === 'report-generation' && (
            <Card>
              <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Report Generation</h2>
              
              <FormGroup>
                <Label>Select Document for Report</Label>
                <Select
                  value={selectedDocument}
                  onChange={(e) => setSelectedDocument(e.target.value)}
                >
                  <option value="">Choose a document...</option>
                  {documents.filter(doc => doc.status === 'analyzed' || doc.status === 'completed').map(doc => (
                    <option key={doc.id} value={doc.id}>
                      {doc.filename} ({doc.status})
                    </option>
                  ))}
                </Select>
              </FormGroup>

              <div style={{background: '#f8fafc', padding: '1.5rem', borderRadius: '8px', margin: '1.5rem 0'}}>
                <h3 style={{color: '#1f2937', marginBottom: '1rem'}}>Report Includes:</h3>
                <ul style={{color: '#6b7280', lineHeight: '1.6'}}>
                  <li>📋 Executive Summary with key insights and recommendations</li>
                  <li>📊 Volume Metrics and trend analysis</li>
                  <li>😊 Detailed Sentiment Analysis breakdown</li>
                  <li>🧍‍♂️ Audience Insights and demographics (where available)</li>
                  <li>🔥 Top Performing Content and engagement metrics</li>
                  <li>🗣️ Key Topics & Hashtags analysis</li>
                  <li>⚔️ Competitor & Industry Benchmarking</li>
                  <li>🚨 Crisis or Issue Tracking (if relevant)</li>
                  <li>💡 Strategic Insights & Recommendations</li>
                  <li>📅 Methodology & Data Sources</li>
                </ul>
              </div>

              <ButtonGroup>
                <Button onClick={generateReport} disabled={!selectedDocument || loading}>
                  <FileText size={16} />
                  Generate PowerPoint Report
                </Button>
                {selectedDocument && (
                  <Button $variant="secondary" onClick={() => {
                    const doc = documents.find(d => d.id === selectedDocument);
                    if (doc) viewDocument(doc.id);
                  }}>
                    <Eye size={16} />
                    Preview Data
                  </Button>
                )}
              </ButtonGroup>
            </Card>
          )}

          {/* User Management */}
          {activeNav === 'user-management' && (
            <Card>
              <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>User Management</h2>
              <p style={{color: '#6b7280'}}>User management functionality would be implemented here for managing system access and permissions.</p>
            </Card>
          )}
        </Content>
      </MainContent>
    </AppContainer>
  );
}

export default App;