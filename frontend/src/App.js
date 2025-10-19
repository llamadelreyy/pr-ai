
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
  RefreshCw
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

const Button = styled.button`
  width: ${props => props.$fullWidth ? '100%' : 'auto'};
  background: ${props => props.$variant === 'secondary' ? 'white' : '#2596be'};
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
    background: ${props => props.$variant === 'secondary' ? '#f8fafc' : '#1e88e5'};
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

const StepIndicator = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 3rem;
  gap: 1rem;
  flex-wrap: wrap;
`;

const Step = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  border-radius: 6px;
  background: ${props => props.$active ? '#2596be' : 'white'};
  color: ${props => props.$active ? 'white' : '#6b7280'};
  border: 2px solid ${props => props.$completed ? '#10b981' : props.$active ? '#2596be' : '#e5e7eb'};
  transition: all 0.3s ease;
  font-size: 0.875rem;
  font-weight: 500;
`;

const StepNumber = styled.span`
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: ${props => props.$completed ? '#10b981' : props.$active ? 'white' : '#f3f4f6'};
  color: ${props => props.$completed ? 'white' : props.$active ? '#2596be' : '#6b7280'};
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 600;
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
  justify-content: between;
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
  margin-left: auto;
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

const ConfidenceContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
`;

const ConfidenceText = styled.span`
  font-size: 0.75rem;
  font-weight: 600;
  color: ${props => {
    if (props.$confidence >= 0.8) return '#166534';
    if (props.$confidence >= 0.6) return '#d97706';
    return '#991b1b';
  }};
`;

const ConfidenceBar = styled.div`
  width: 100%;
  height: 8px;
  background: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
`;

const ConfidenceFill = styled.div`
  height: 100%;
  background: ${props => {
    if (props.$confidence >= 0.8) return '#10b981';
    if (props.$confidence >= 0.6) return '#f59e0b';
    return '#ef4444';
  }};
  width: ${props => props.$confidence * 100}%;
  transition: width 0.3s ease;
`;

const Stats = styled.div`
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

const API_BASE = 'http://localhost:8011';

function App() {
  // Authentication state
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loginForm, setLoginForm] = useState({ username: '', password: '' });
  const [loginError, setLoginError] = useState('');

  // UI state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeNav, setActiveNav] = useState('dashboard');
  
  // Application state
  const [currentStep, setCurrentStep] = useState(1);
  const [sessionId, setSessionId] = useState(null);
  const [uploadedData, setUploadedData] = useState(null);
  const [analyzedData, setAnalyzedData] = useState(null);
  const [finalData, setFinalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [topicStats, setTopicStats] = useState(null);
  const [averageConfidence, setAverageConfidence] = useState(null);
  const [streamingMode, setStreamingMode] = useState(false);
  const [streamingData, setStreamingData] = useState([]);
  const [streamingStatus, setStreamingStatus] = useState("");
  const [analysisStreamingMode, setAnalysisStreamingMode] = useState(false);
  const [analysisStreamingData, setAnalysisStreamingData] = useState([]);
  const [analysisStreamingStatus, setAnalysisStreamingStatus] = useState("");

  // Check for saved authentication on mount
  useEffect(() => {
    const savedAuth = localStorage.getItem('meioAuth');
    if (savedAuth === 'true') {
      setIsAuthenticated(true);
    }
  }, []);

  const handleLogin = (e) => {
    e.preventDefault();
    setLoginError('');
    
    // Dummy credentials
    if (loginForm.username === 'admin' && loginForm.password === 'meio2024') {
      setIsAuthenticated(true);
      localStorage.setItem('meioAuth', 'true');
    } else {
      setLoginError('Invalid credentials. Use admin/meio2024');
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    localStorage.removeItem('meioAuth');
    setCurrentStep(1);
    setSessionId(null);
    setUploadedData(null);
    setAnalyzedData(null);
    setFinalData(null);
    setStats(null);
  };

  const onDrop = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setSessionId(response.data.session_id);
      setUploadedData(response.data);
      setCurrentStep(2);
    } catch (error) {
      alert('Error uploading file: ' + (error.response?.data?.detail || error.message));
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

  const viewData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE}/data/${sessionId}`);
      setUploadedData({...uploadedData, data: response.data.data});
      setCurrentStep(3);
    } catch (error) {
      alert('Error loading data: ' + (error.response?.data?.detail || error.message));
    }
    setLoading(false);
  };

  const analyzeDataStreaming = async () => {
    setAnalysisStreamingMode(true);
    setAnalysisStreamingData([]);
    setAnalysisStreamingStatus("Starting sentiment analysis...");
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE}/analyze-streaming`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId
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
                      ? {...item, topics: extractDetailedTopics(data.topic)}
                      : item
                  ));
                  break;
                  
                case 'row_complete':
                  setAnalysisStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, sentiment: data.sentiment, confidence: data.confidence, topics: extractDetailedTopics(data.topic), complete: true}
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
                  setAnalyzedData(data.analyzed_data);
                  setStats(data.statistics);
                  setTopicStats(data.topic_statistics);
                  setAverageConfidence(data.average_confidence);
                  setAnalysisStreamingStatus("Sentiment analysis completed!");
                  setLoading(false);
                  setTimeout(() => {
                    setAnalysisStreamingMode(false);
                    setCurrentStep(4);
                  }, 2000);
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
      setLoading(false);
      setAnalysisStreamingMode(false);
    }
  };

  const generateCommentsStreaming = async () => {
    setStreamingMode(true);
    setStreamingData([]);
    setStreamingStatus("Starting comment generation...");
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE}/mitigate-streaming`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId
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
                  setStreamingStatus(`Processing row ${data.index + 1}/${data.total}: ${data.text}...`);
                  setStreamingData(prev => [...prev, {
                    index: data.index,
                    text: data.text,
                    sentiment: data.sentiment,
                    topic: data.topic,
                    comment: "",
                    complete: false
                  }]);
                  break;
                  
                case 'word':
                  setStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, comment: data.complete}
                      : item
                  ));
                  break;
                  
                case 'complete':
                  setStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, comment: data.comment, complete: true}
                      : item
                  ));
                  break;
                  
                case 'row_complete':
                  setStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, comment: data.comment, complete: true}
                      : item
                  ));
                  break;
                  
                case 'error':
                  setStreamingData(prev => prev.map(item =>
                    item.index === data.index
                      ? {...item, comment: `Error: ${data.error}`, complete: true}
                      : item
                  ));
                  break;
                  
                case 'complete_all':
                  setFinalData(data.data);
                  setStreamingStatus("Comment generation completed!");
                  setLoading(false);
                  setTimeout(() => {
                    setStreamingMode(false);
                    setCurrentStep(5);
                  }, 2000);
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
      setStreamingStatus("Error occurred during streaming: " + error.message);
      setLoading(false);
      setStreamingMode(false);
    }
  };

  const generateReport = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/generate-report`, {
        session_id: sessionId
      }, {
        responseType: 'blob'
      });
      
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `MEIO_Sentiment_Report_${new Date().toISOString().slice(0,10)}.pptx`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      setCurrentStep(6);
    } catch (error) {
      alert('Error generating report: ' + (error.response?.data?.detail || error.message));
    }
    setLoading(false);
  };

  const getSentimentIcon = (sentiment) => {
    switch(sentiment) {
      case 'positive': return <Smile size={14} />;
      case 'negative': return <Frown size={14} />;
      case 'neutral': return <Minus size={14} />;
      default: return <Minus size={14} />;
    }
  };

  const resetWorkflow = () => {
    setCurrentStep(1);
    setSessionId(null);
    setUploadedData(null);
    setAnalyzedData(null);
    setFinalData(null);
    setStats(null);
    setActiveNav('dashboard');
  };

  // Enhanced topic extraction function
  const extractDetailedTopics = (topicString) => {
    if (!topicString) return ['General'];
    
    // Split by common separators and clean up
    const topics = topicString
      .split(/[,;|&+]/)
      .map(topic => topic.trim())
      .filter(topic => topic.length > 0)
      .slice(0, 4); // Limit to 4 topics max
    
    return topics.length > 0 ? topics : ['General'];
  };

  // Enhanced confidence calculation
  const calculateEnhancedConfidence = (confidence, textLength = 50, sentimentStrength = 1) => {
    let baseConfidence = parseFloat(confidence) || 0.5;
    
    // Adjust based on text length (longer texts might have more reliable analysis)
    const lengthFactor = Math.min(textLength / 100, 1.2);
    
    // Adjust based on sentiment strength (stronger sentiments might be more confident)
    const strengthFactor = sentimentStrength * 0.1;
    
    // Calculate enhanced confidence
    let enhancedConfidence = baseConfidence * lengthFactor + strengthFactor;
    
    // Ensure confidence stays within bounds
    enhancedConfidence = Math.max(0.1, Math.min(1.0, enhancedConfidence));
    
    return enhancedConfidence;
  };

  const getPageTitle = () => {
    switch (activeNav) {
      case 'dashboard': return 'Sentiment Analysis Dashboard';
      case 'data': return 'Data Management';
      case 'analysis': return 'Analysis Results';
      case 'settings': return 'System Settings';
      default: return 'Sentiment Analysis System';
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
            $active={activeNav === 'data'}
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('data')}
          >
            <Database size={20} />
            <span>Data Upload</span>
          </NavItem>
          
          <NavItem
            $active={activeNav === 'analysis'}
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('analysis')}
          >
            <Activity size={20} />
            <span>Analysis</span>
          </NavItem>
          
          <NavItem
            $active={activeNav === 'users'}
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('users')}
          >
            <Users size={20} />
            <span>Users</span>
          </NavItem>
          
          <NavItem
            $active={activeNav === 'settings'}
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('settings')}
          >
            <Settings size={20} />
            <span>Settings</span>
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
          {(activeNav === 'dashboard' || activeNav === 'data' || activeNav === 'analysis') && (
            <>
              <StepIndicator>
                <Step $active={currentStep === 1} $completed={currentStep > 1}>
                  <StepNumber $active={currentStep === 1} $completed={currentStep > 1}>
                    {currentStep > 1 ? <CheckCircle size={16} /> : '1'}
                  </StepNumber>
                  Upload Data
                </Step>
                <Step $active={currentStep === 2} $completed={currentStep > 2}>
                  <StepNumber $active={currentStep === 2} $completed={currentStep > 2}>
                    {currentStep > 2 ? <CheckCircle size={16} /> : '2'}
                  </StepNumber>
                  View Data
                </Step>
                <Step $active={currentStep === 3} $completed={currentStep > 3}>
                  <StepNumber $active={currentStep === 3} $completed={currentStep > 3}>
                    {currentStep > 3 ? <CheckCircle size={16} /> : '3'}
                  </StepNumber>
                  Analyze
                </Step>
                <Step $active={currentStep === 4} $completed={currentStep > 4}>
                  <StepNumber $active={currentStep === 4} $completed={currentStep > 4}>
                    {currentStep > 4 ? <CheckCircle size={16} /> : '4'}
                  </StepNumber>
                  Generate Reports
                </Step>
                <Step $active={currentStep === 5} $completed={currentStep > 5}>
                  <StepNumber $active={currentStep === 5} $completed={currentStep > 5}>
                    {currentStep > 5 ? <CheckCircle size={16} /> : '5'}
                  </StepNumber>
                  Download
                </Step>
              </StepIndicator>

              {loading && (
                <LoadingSpinner>
                  <Loader className="animate-spin" size={24} />
                  Processing...
                </LoadingSpinner>
              )}

              {/* Step 1: Upload */}
              {currentStep === 1 && (
                <Card>
                  <h2 style={{marginBottom: '1.5rem', textAlign: 'center', color: '#1f2937'}}>Upload Data File</h2>
                  <DropZone {...getRootProps()} $isDragActive={isDragActive}>
                    <input {...getInputProps()} />
                    <Upload size={48} style={{margin: '0 auto 1rem', color: '#2596be'}} />
                    <h3 style={{color: '#1f2937', marginBottom: '0.5rem'}}>Drop your CSV or XLSX file here</h3>
                    <p style={{color: '#6b7280', fontSize: '0.875rem'}}>
                      Or click to select file (Max 13,000 rows supported)
                    </p>
                  </DropZone>
                </Card>
              )}

              {/* Step 2: View Data */}
              {currentStep === 2 && uploadedData && (
                <Card>
                  <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Data Preview</h2>
                  <div style={{marginBottom: '1rem', color: '#6b7280'}}>
                    <strong>File:</strong> {uploadedData.filename} |
                    <strong> Rows:</strong> {uploadedData.row_count} |
                    <strong> Columns:</strong> {uploadedData.columns?.join(', ')}
                  </div>
                  
                  {uploadedData.preview && (
                    <TableContainer>
                      <TableHeader>
                        <TableTitle>Data Preview (First 5 rows)</TableTitle>
                      </TableHeader>
                      <Table>
                        <thead>
                          <tr>
                            {uploadedData.columns?.map(col => (
                              <Th key={col}>{col}</Th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {uploadedData.preview.map((row, idx) => (
                            <tr key={idx}>
                              {uploadedData.columns?.map(col => (
                                <Td key={col} title={row[col]}>
                                  {String(row[col] || '').slice(0, 50)}
                                  {String(row[col] || '').length > 50 ? '...' : ''}
                                </Td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </TableContainer>
                  )}
                  
                  <ButtonGroup>
                    <Button $variant="secondary" onClick={resetWorkflow}>
                      <Upload size={16} />
                      Upload New File
                    </Button>
                    <Button onClick={viewData} disabled={loading}>
                      <Eye size={16} />
                      View Full Data
                    </Button>
                  </ButtonGroup>
                </Card>
              )}

              {/* Step 3: Analyze */}
              {currentStep === 3 && (
                <Card>
                  <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Ready for Sentiment Analysis</h2>
                  <p style={{textAlign: 'center', marginBottom: '2rem', color: '#6b7280'}}>
                    Click "Analyze" to perform AI-powered sentiment analysis on your data.
                    This process may take a few minutes for large datasets.
                  </p>
                  
                  <ButtonGroup>
                    <Button $variant="secondary" onClick={() => setCurrentStep(2)}>
                      <Eye size={16} />
                      Back to Data
                    </Button>
                    <Button onClick={analyzeDataStreaming} disabled={loading} style={{background: '#10b981'}}>
                      <TrendingUp size={16} />
                      Start Analysis
                    </Button>
                  </ButtonGroup>
                </Card>
              )}

              {/* Analysis Streaming Interface */}
              {analysisStreamingMode && (
                <Card>
                  <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Real-time Sentiment Analysis</h2>
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
                            <ConfidenceContainer>
                              <ConfidenceText $confidence={item.confidence}>
                                {(item.confidence * 100).toFixed(1)}%
                              </ConfidenceText>
                              <ConfidenceBar>
                                <ConfidenceFill $confidence={item.confidence} />
                              </ConfidenceBar>
                            </ConfidenceContainer>
                          )}
                          {item.topics && item.topics.length > 0 && (
                            <TagsContainer>
                              {item.topics.map((topic, idx) => (
                                <TopicBadge key={idx}>
                                  <Tag size={12} />
                                  {topic}
                                </TopicBadge>
                              ))}
                            </TagsContainer>
                          )}
                          {!item.complete && (
                            <span style={{color: '#10b981', fontSize: '0.875rem'}}>Analyzing...</span>
                          )}
                        </div>
                      </StreamingRow>
                    ))}
                  </StreamingContainer>
                </Card>
              )}

              {/* Step 4: Analysis Results */}
              {currentStep === 4 && analyzedData && stats && !analysisStreamingMode && (
                <Card>
                  <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Sentiment Analysis Results</h2>
                  
                  <Stats>
                    <StatCard>
                      <StatNumber color="#10b981">{stats.positive || 0}</StatNumber>
                      <StatLabel>Positive ({analyzedData.length > 0 ? (((stats.positive || 0) / analyzedData.length) * 100).toFixed(1) : '0.0'}%)</StatLabel>
                    </StatCard>
                    <StatCard>
                      <StatNumber color="#ef4444">{stats.negative || 0}</StatNumber>
                      <StatLabel>Negative ({analyzedData.length > 0 ? (((stats.negative || 0) / analyzedData.length) * 100).toFixed(1) : '0.0'}%)</StatLabel>
                    </StatCard>
                    <StatCard>
                      <StatNumber color="#6b7280">{stats.neutral || 0}</StatNumber>
                      <StatLabel>Neutral ({analyzedData.length > 0 ? (((stats.neutral || 0) / analyzedData.length) * 100).toFixed(1) : '0.0'}%)</StatLabel>
                    </StatCard>
                    <StatCard>
                      <StatNumber>{averageConfidence && !isNaN(averageConfidence) ? (averageConfidence * 100).toFixed(1) : '0.0'}%</StatNumber>
                      <StatLabel>Average Confidence</StatLabel>
                    </StatCard>
                  </Stats>

                  {topicStats && (
                    <Card style={{marginBottom: '1rem'}}>
                      <h3 style={{marginBottom: '1rem', color: '#1f2937'}}>Topic Distribution</h3>
                      <Stats>
                        {Object.entries(topicStats).slice(0, 4).map(([topic, count]) => (
                          <StatCard key={topic}>
                            <StatNumber color="#2596be">{count}</StatNumber>
                            <StatLabel>{topic}</StatLabel>
                          </StatCard>
                        ))}
                      </Stats>
                    </Card>
                  )}

                  <TableContainer>
                    <TableHeader>
                      <TableTitle>Analysis Results</TableTitle>
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
                          <Th>Content</Th>
                          <Th>Sentiment</Th>
                          <Th>Confidence</Th>
                          <Th>Topics</Th>
                          <Th>Tags</Th>
                        </tr>
                      </thead>
                      <tbody>
                        {analyzedData.slice(0, 100).map((row, idx) => {
                          const textContent = Object.values(row).find(val =>
                            typeof val === 'string' && val.length > 10 &&
                            !['sentiment', 'confidence', 'topic'].includes(val)
                          ) || 'No text content';
                          
                          const topics = extractDetailedTopics(row.topic);
                          const enhancedConfidence = calculateEnhancedConfidence(
                            row.confidence,
                            String(textContent).length,
                            row.sentiment === 'positive' ? 1.2 : row.sentiment === 'negative' ? 1.1 : 1.0
                          );
                          
                          return (
                            <tr key={idx}>
                              <ContentCell>
                                {String(textContent)}
                              </ContentCell>
                              <Td>
                                <SentimentBadge $sentiment={row.sentiment}>
                                  {getSentimentIcon(row.sentiment)}
                                  {row.sentiment}
                                </SentimentBadge>
                              </Td>
                              <Td>
                                <ConfidenceContainer>
                                  <ConfidenceText $confidence={enhancedConfidence}>
                                    {(enhancedConfidence * 100).toFixed(1)}%
                                  </ConfidenceText>
                                  <ConfidenceBar>
                                    <ConfidenceFill $confidence={enhancedConfidence} />
                                  </ConfidenceBar>
                                </ConfidenceContainer>
                              </Td>
                              <Td>
                                <TagsContainer>
                                  {topics.map((topic, topicIdx) => (
                                    <TopicBadge key={topicIdx}>
                                      <Tag size={12} />
                                      {topic}
                                    </TopicBadge>
                                  ))}
                                </TagsContainer>
                              </Td>
                              <Td>
                                <TagsContainer>
                                  <TopicBadge style={{background: '#fef3c7', color: '#92400e', borderColor: '#fbbf24'}}>
                                    <Activity size={12} />
                                    {row.sentiment === 'positive' ? 'Boost' : row.sentiment === 'negative' ? 'Monitor' : 'Neutral'}
                                  </TopicBadge>
                                  {enhancedConfidence > 0.8 && (
                                    <TopicBadge style={{background: '#dcfce7', color: '#166534', borderColor: '#22c55e'}}>
                                      <CheckCircle size={12} />
                                      High Confidence
                                    </TopicBadge>
                                  )}
                                </TagsContainer>
                              </Td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </Table>
                  </TableContainer>
                  
                  <ButtonGroup>
                    <Button $variant="secondary" onClick={() => setCurrentStep(3)}>
                      <BarChart3 size={16} />
                      Re-analyze
                    </Button>
                    <Button onClick={generateCommentsStreaming} disabled={loading} style={{background: '#10b981'}}>
                      <MessageSquare size={16} />
                      Generate Comments
                    </Button>
                  </ButtonGroup>
                </Card>
              )}

              {/* Streaming Interface for Comments */}
              {streamingMode && (
                <Card>
                  <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Real-time Comment Generation</h2>
                  <div style={{marginBottom: '1rem', fontWeight: 'bold', color: '#10b981'}}>
                    {streamingStatus}
                  </div>
                  
                  <StreamingContainer>
                    {streamingData.map((item, index) => (
                      <StreamingRow key={index}>
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
                </Card>
              )}

              {/* Step 5: Comments Generated */}
              {currentStep === 5 && finalData && !streamingMode && (
                <Card>
                  <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Mitigation Strategy Generated</h2>
                  <p style={{textAlign: 'center', marginBottom: '2rem', color: '#6b7280'}}>
                    Counter-comments for negative sentiments, amplification for positive sentiments,
                    and positive-leaning responses for neutral sentiments have been generated.
                  </p>
                  
                  <TableContainer>
                    <TableHeader>
                      <TableTitle>Generated Comments</TableTitle>
                    </TableHeader>
                    <Table>
                      <thead>
                        <tr>
                          <Th>Original</Th>
                          <Th>Sentiment</Th>
                          <Th>Confidence</Th>
                          <Th>Topics</Th>
                          <Th>Generated Response</Th>
                        </tr>
                      </thead>
                      <tbody>
                        {finalData.slice(0, 50).map((row, idx) => {
                          const textContent = Object.values(row).find(val =>
                            typeof val === 'string' && val.length > 10 &&
                            !['sentiment', 'confidence', 'topic', 'generated_comment'].includes(val)
                          ) || 'No text content';
                          
                          const topics = extractDetailedTopics(row.topic);
                          const enhancedConfidence = calculateEnhancedConfidence(row.confidence, String(textContent).length);
                          
                          return (
                            <tr key={idx}>
                              <ContentCell>
                                {String(textContent)}
                              </ContentCell>
                              <Td>
                                <SentimentBadge $sentiment={row.sentiment}>
                                  {getSentimentIcon(row.sentiment)}
                                  {row.sentiment}
                                </SentimentBadge>
                              </Td>
                              <Td>
                                <ConfidenceContainer>
                                  <ConfidenceText $confidence={enhancedConfidence}>
                                    {(enhancedConfidence * 100).toFixed(1)}%
                                  </ConfidenceText>
                                  <ConfidenceBar>
                                    <ConfidenceFill $confidence={enhancedConfidence} />
                                  </ConfidenceBar>
                                </ConfidenceContainer>
                              </Td>
                              <Td>
                                <TagsContainer>
                                  {topics.map((topic, topicIdx) => (
                                    <TopicBadge key={topicIdx}>
                                      <Tag size={12} />
                                      {topic}
                                    </TopicBadge>
                                  ))}
                                </TagsContainer>
                              </Td>
                              <ContentCell>
                                {String(row.generated_comment || '')}
                              </ContentCell>
                            </tr>
                          );
                        })}
                      </tbody>
                    </Table>
                  </TableContainer>
                  
                  <ButtonGroup>
                    <Button $variant="secondary" onClick={() => setCurrentStep(4)}>
                      <MessageSquare size={16} />
                      Back to Analysis
                    </Button>
                    <Button onClick={generateReport} disabled={loading}>
                      <FileText size={16} />
                      Generate Report
                    </Button>
                  </ButtonGroup>
                </Card>
              )}

              {/* Step 6: Report Generated */}
              {currentStep === 6 && (
                <Card>
                  <div style={{textAlign: 'center'}}>
                    <CheckCircle size={64} style={{color: '#10b981', margin: '0 auto 1rem'}} />
                    <h2 style={{marginBottom: '1rem', color: '#1f2937'}}>Report Generated Successfully</h2>
                    <p style={{color: '#6b7280', marginBottom: '2rem'}}>
                      Your comprehensive sentiment analysis report has been downloaded as a PowerPoint presentation.
                    </p>
                    
                    <ButtonGroup>
                      <Button onClick={resetWorkflow}>
                        <Upload size={16} />
                        Start New Analysis
                      </Button>
                      <Button $variant="secondary" onClick={generateReport} disabled={loading}>
                        <Download size={16} />
                        Download Again
                      </Button>
                    </ButtonGroup>
                  </div>
                </Card>
              )}
            </>
          )}

          {/* Other Navigation Pages */}
          {activeNav === 'users' && (
            <Card>
              <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>User Management</h2>
              <p style={{color: '#6b7280'}}>User management functionality would be implemented here.</p>
            </Card>
          )}

          {activeNav === 'settings' && (
            <Card>
              <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>System Settings</h2>
              <p style={{color: '#6b7280'}}>System configuration options would be available here.</p>
            </Card>
          )}
        </Content>
      </MainContent>
    </AppContainer>
  );
}

export default App;