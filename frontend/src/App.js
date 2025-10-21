
import React, { useState, useEffect, useRef } from 'react';
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
  ArrowLeft,
  Target,
  Crown,
  Zap,
  FilterX,
  SortAsc,
  SortDesc,
  ExternalLink,
  Globe,
  User,
  Building,
  Bot
} from 'lucide-react';

// Styled Components
const AppContainer = styled.div`
  height: 100vh;
  background: #ffffff;
  color: #1f2937;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  display: flex;
  overflow: hidden;
`;

const Sidebar = styled.div`
  width: ${props => props.$collapsed ? '60px' : '280px'};
  background: #2596be;
  color: white;
  transition: width 0.3s ease;
  display: flex;
  flex-direction: column;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
  flex-shrink: 0;
  height: 100vh;
  overflow-y: auto;
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
  height: 100vh;
  overflow: hidden;
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
  height: calc(100vh - 80px); /* Subtract header height */
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
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
  margin: 2rem auto;
  max-width: 1200px;
  max-height: 600px;
  overflow: auto;
  display: flex;
  flex-direction: column;
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
  min-width: 800px;
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

const ProgressContainer = styled.div`
  margin: 1rem 0;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
  margin: 0.5rem 0;
`;

const ProgressFill = styled.div`
  height: 100%;
  background: #2596be;
  border-radius: 4px;
  transition: width 0.3s ease;
  width: ${props => props.$progress}%;
`;

const ProgressText = styled.div`
  font-size: 0.875rem;
  color: #374151;
  font-weight: 500;
  text-align: center;
`;

const HandsontableContainer = styled.div`
  width: 100%;
  height: 600px;
  margin: 1rem 0;
  
  .handsontable {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 0.875rem;
  }
  
  .ht_master .wtHolder {
    background: white;
  }
  
  .handsontable th {
    background: #f8fafc;
    font-weight: 600;
    color: #374151;
    border-bottom: 1px solid #e5e7eb;
  }
  
  .handsontable td {
    border-color: #e5e7eb;
    color: #374151;
  }
  
  .handsontable .currentRow {
    background: #f0f9ff;
  }
  
  .handsontable .area-1 {
    background: #dbeafe;
  }
`;

// Simple Chart Components for Visualizations
const ChartContainer = styled.div`
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
  margin-bottom: 1rem;
`;

const ChartTitle = styled.h4`
  color: #1f2937;
  margin-bottom: 1rem;
  font-size: 1rem;
  font-weight: 600;
`;

const BarChart = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin: 1rem 0;
`;

const BarItem = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const BarLabel = styled.div`
  min-width: 100px;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
`;

const BarContainer = styled.div`
  flex: 1;
  height: 24px;
  background: #f3f4f6;
  border-radius: 12px;
  overflow: hidden;
  position: relative;
`;

const BarFill = styled.div`
  height: 100%;
  background: ${props => props.color || '#2596be'};
  border-radius: 12px;
  transition: width 0.3s ease;
  width: ${props => props.percentage}%;
`;

const BarValue = styled.div`
  min-width: 60px;
  text-align: right;
  font-size: 0.875rem;
  color: #6b7280;
  font-weight: 500;
`;

const PieChartContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 2rem;
  margin: 1rem 0;
`;

const PieChart = styled.div`
  width: 120px;
  height: 120px;
  border-radius: 50%;
  background: conic-gradient(
    ${props => props.colors}
  );
  position: relative;
`;

const PieCenter = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 60px;
  height: 60px;
  background: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 600;
  color: #374151;
`;

const LegendContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const LegendItem = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
`;

const LegendColor = styled.div`
  width: 12px;
  height: 12px;
  border-radius: 2px;
  background: ${props => props.color};
`;

// Fast Handsontable Component for large datasets
const FastDataTable = ({ data, columns, title }) => {
  const containerRef = useRef(null);
  const hotRef = useRef(null);

  useEffect(() => {
    if (!data || !columns || !containerRef.current) return;

    // Destroy existing instance
    if (hotRef.current) {
      hotRef.current.destroy();
    }

    // Create new Handsontable instance
    hotRef.current = new window.Handsontable(containerRef.current, {
      data: data,
      colHeaders: columns,
      rowHeaders: true,
      columnSorting: true,
      manualColumnResize: true,
      manualRowResize: true,
      contextMenu: true,
      copyPaste: true,
      search: true,
      filters: true,
      dropdownMenu: true,
      width: '100%',
      height: '100%',
      licenseKey: 'non-commercial-and-evaluation',
      stretchH: 'all',
      autoWrapRow: true,
      autoWrapCol: true,
      renderAllRows: false, // Enable virtualization for performance
      viewportRowRenderingOffset: 100,
      outsideClickDeselects: false,
      selectionMode: 'multiple',
      readOnly: true,
      className: 'fast-data-table',
      cells: function (row, col) {
        const cellProperties = {};
        const cellData = this.instance.getDataAtCell(row, col);
        
        // Special rendering for sentiment columns
        if (columns[col]?.toLowerCase().includes('sentiment')) {
          cellProperties.renderer = sentimentRenderer;
        }
        // Special rendering for confidence columns
        else if (columns[col]?.toLowerCase().includes('confidence')) {
          cellProperties.renderer = confidenceRenderer;
        }
        // Special rendering for topic columns
        else if (columns[col]?.toLowerCase().includes('topic')) {
          cellProperties.renderer = topicRenderer;
        }
        // Text wrapping for long content
        else if (typeof cellData === 'string' && cellData.length > 50) {
          cellProperties.renderer = textRenderer;
        }
        
        return cellProperties;
      }
    });

    // Cleanup on unmount
    return () => {
      if (hotRef.current) {
        hotRef.current.destroy();
        hotRef.current = null;
      }
    };
  }, [data, columns]);

  // Custom renderers for different data types - Fixed with proper type checking
  const sentimentRenderer = function(instance, td, row, col, prop, value, cellProperties) {
    window.Handsontable.renderers.TextRenderer.call(this, instance, td, row, col, prop, value, cellProperties);
    
    if (value && typeof value === 'string' && value.trim()) {
      const sentiment = String(value).toLowerCase().trim();
      td.style.padding = '8px';
      
      if (sentiment === 'positive') {
        td.style.backgroundColor = '#dcfce7';
        td.style.color = '#166534';
        td.style.fontWeight = '600';
        td.innerHTML = `<span>😊 ${value}</span>`;
      } else if (sentiment === 'negative') {
        td.style.backgroundColor = '#fee2e2';
        td.style.color = '#991b1b';
        td.style.fontWeight = '600';
        td.innerHTML = `<span>😔 ${value}</span>`;
      } else if (sentiment === 'neutral') {
        td.style.backgroundColor = '#f3f4f6';
        td.style.color = '#374151';
        td.style.fontWeight = '600';
        td.innerHTML = `<span>😐 ${value}</span>`;
      }
    }
  };

  const confidenceRenderer = function(instance, td, row, col, prop, value, cellProperties) {
    window.Handsontable.renderers.TextRenderer.call(this, instance, td, row, col, prop, value, cellProperties);
    
    if (value !== null && value !== undefined && value !== '') {
      const confidence = parseFloat(String(value));
      if (!isNaN(confidence)) {
        const percentage = confidence > 1 ? confidence : confidence * 100;
        td.style.padding = '8px';
        td.style.fontWeight = '500';
        
        if (percentage >= 80) {
          td.style.color = '#166534';
          td.style.backgroundColor = '#dcfce7';
        } else if (percentage >= 60) {
          td.style.color = '#ea580c';
          td.style.backgroundColor = '#fed7aa';
        } else {
          td.style.color = '#991b1b';
          td.style.backgroundColor = '#fee2e2';
        }
        
        td.innerHTML = `${percentage.toFixed(1)}%`;
      }
    }
  };

  const topicRenderer = function(instance, td, row, col, prop, value, cellProperties) {
    window.Handsontable.renderers.TextRenderer.call(this, instance, td, row, col, prop, value, cellProperties);
    
    if (value && typeof value === 'string' && value.trim()) {
      td.style.padding = '8px';
      const topics = String(value).split(',').map(t => t.trim()).filter(t => t).slice(0, 3);
      if (topics.length > 0) {
        const topicBadges = topics.map(topic =>
          `<span style="display: inline-block; background: #eff6ff; color: #1e40af; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin: 1px; border: 1px solid #bfdbfe;">${topic}</span>`
        ).join(' ');
        td.innerHTML = topicBadges;
      }
    }
  };

  const textRenderer = function(instance, td, row, col, prop, value, cellProperties) {
    window.Handsontable.renderers.TextRenderer.call(this, instance, td, row, col, prop, value, cellProperties);
    
    if (value && typeof value === 'string' && value.trim()) {
      td.style.padding = '8px';
      td.style.lineHeight = '1.4';
      td.style.maxWidth = '400px';
      td.style.whiteSpace = 'normal';
      td.style.overflow = 'hidden';
      td.style.textOverflow = 'ellipsis';
      
      // Truncate very long text for performance
      const textValue = String(value);
      if (textValue.length > 200) {
        td.innerHTML = textValue.substring(0, 200) + '...';
        td.title = textValue; // Show full text on hover
      } else {
        td.innerHTML = textValue;
      }
    }
  };

  return (
    <TableContainer>
      <TableHeader style={{flexShrink: 0}}>
        <TableTitle>{title}</TableTitle>
        <TableActions>
          <Button $variant="secondary" style={{padding: '0.5rem'}} onClick={() => {
            if (hotRef.current) {
              hotRef.current.getPlugin('search').query('');
            }
          }}>
            <Search size={16} />
          </Button>
        </TableActions>
      </TableHeader>
      <HandsontableContainer>
        <div ref={containerRef} style={{width: '100%', height: '100%'}} />
      </HandsontableContainer>
    </TableContainer>
  );
};

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
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState('');
  const [currentDocumentData, setCurrentDocumentData] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [responses, setResponses] = useState(null);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState(''); // 'success' or 'error'
  const [selectedTextColumn, setSelectedTextColumn] = useState('');
  const [detectedTextColumn, setDetectedTextColumn] = useState('');
  
  // Streaming states
  const [analysisStreamingMode, setAnalysisStreamingMode] = useState(false);
  const [analysisStreamingData, setAnalysisStreamingData] = useState([]);
  const [analysisStreamingStatus, setAnalysisStreamingStatus] = useState("");
  const [responseStreamingMode, setResponseStreamingMode] = useState(false);
  const [responseStreamingData, setResponseStreamingData] = useState([]);
  const [responseStreamingStatus, setResponseStreamingStatus] = useState("");
  
  // Influential voices analysis states
  const [influentialVoices, setInfluentialVoices] = useState(null);
  const [exposureThreshold, setExposureThreshold] = useState(500000);
  const [topCount, setTopCount] = useState(5);
  const [useThreshold, setUseThreshold] = useState(true);
  const [filterSentiment, setFilterSentiment] = useState('all');
  const [sortBy, setSortBy] = useState('exposure');
  const [sortOrder, setSortOrder] = useState('desc');
  
  // Saved analyses states
  const [savedAnalyses, setSavedAnalyses] = useState([]);
  const [selectedSavedAnalysis, setSelectedSavedAnalysis] = useState('');

  // Check for saved authentication on mount
  useEffect(() => {
    const savedAuth = localStorage.getItem('meioAuth');
    if (savedAuth === 'true') {
      setIsAuthenticated(true);
      loadDocuments();
      loadSavedAnalyses();
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

  const loadSavedAnalyses = async () => {
    try {
      const response = await axios.get(`${FINAL_API_BASE}/saved-analyses`);
      setSavedAnalyses(response.data.saved_analyses || []);
    } catch (error) {
      console.error('Error loading saved analyses:', error);
    }
  };

  const saveAnalysisSession = async (analysisName, analysisType, analysisData) => {
    if (!selectedDocument) {
      showMessage('Please select a document first', 'error');
      return;
    }

    try {
      const response = await axios.post(`${FINAL_API_BASE}/save-analysis`, {
        session_id: `analysis_${Date.now()}`,
        document_id: selectedDocument,
        analysis_name: analysisName,
        analysis_type: analysisType,
        analysis_data: analysisData
      });
      
      showMessage(`Analysis "${analysisName}" saved successfully!`);
      loadSavedAnalyses();
      return response.data.saved_analysis_id;
    } catch (error) {
      showMessage('Error saving analysis: ' + (error.response?.data?.detail || error.message), 'error');
      return null;
    }
  };

  const loadSavedAnalysis = async (analysisId) => {
    try {
      const response = await axios.get(`${FINAL_API_BASE}/saved-analyses/${analysisId}`);
      const savedAnalysis = response.data;
      
      // Load the analysis data based on type
      if (savedAnalysis.analysis_type === 'sentiment') {
        setAnalysisResults(savedAnalysis.analysis_data);
      } else if (savedAnalysis.analysis_type === 'influential_voices') {
        setInfluentialVoices(savedAnalysis.analysis_data);
      } else if (savedAnalysis.analysis_type === 'responses') {
        setResponses(savedAnalysis.analysis_data);
      }
      
      showMessage(`Loaded saved analysis: ${savedAnalysis.analysis_name}`);
    } catch (error) {
      showMessage('Error loading saved analysis: ' + (error.response?.data?.detail || error.message), 'error');
    }
  };

  const deleteSavedAnalysis = async (analysisId) => {
    if (!window.confirm('Are you sure you want to delete this saved analysis?')) return;
    
    try {
      await axios.delete(`${FINAL_API_BASE}/saved-analyses/${analysisId}`);
      showMessage('Saved analysis deleted successfully!');
      loadSavedAnalyses();
      if (selectedSavedAnalysis === analysisId) {
        setSelectedSavedAnalysis('');
      }
    } catch (error) {
      showMessage('Error deleting saved analysis: ' + (error.response?.data?.detail || error.message), 'error');
    }
  };

  const handleLogin = (e) => {
    e.preventDefault();
    setLoginError('');
    
    if (loginForm.username === 'admin' && loginForm.password === 'meio2024') {
      setIsAuthenticated(true);
      localStorage.setItem('meioAuth', 'true');
      loadDocuments();
      loadSavedAnalyses();
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
    setUploadProgress(0);
    setUploadStatus('Preparing upload...');
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      setUploadStatus('Uploading file...');
      const response = await axios.post(`${FINAL_API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
          setUploadStatus(`Uploading... ${percentCompleted}%`);
        }
      });

      setUploadStatus('Processing file...');

      setUploadStatus('Upload completed!');
      showMessage(`Document "${file.name}" uploaded successfully!`);
      
      // Store the session data with session_id from backend
      const sessionData = {
        id: response.data.session_id,
        session_id: response.data.session_id, // Keep session_id for backend calls
        filename: response.data.filename,
        row_count: response.data.row_count,
        upload_date: new Date().toISOString(),
        status: 'uploaded',
        detected_text_column: response.data.detected_text_column
      };
      
      console.log('Storing session data:', sessionData);
      
      const existingSessions = JSON.parse(localStorage.getItem('meioSessions') || '[]');
      existingSessions.push(sessionData);
      localStorage.setItem('meioSessions', JSON.stringify(existingSessions));
      
      loadDocuments();
    } catch (error) {
      console.error('Upload error details:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown upload error';
      setUploadStatus(`Upload failed: ${errorMessage}`);
      showMessage(`Error uploading file: ${errorMessage}`, 'error');
    }
    
    // Reset upload states
    setLoading(false);
    setTimeout(() => {
      setUploadProgress(0);
      setUploadStatus('');
    }, 3000);
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
    console.log('ViewDocument called with ID:', documentId);
    
    if (!documentId) {
      showMessage('Invalid document ID', 'error');
      return;
    }
    
    setLoading(true);
    try {
      const response = await axios.get(`${FINAL_API_BASE}/data/${documentId}`);
      
      // Find the document in our stored sessions to get detected text column
      const existingSessions = JSON.parse(localStorage.getItem('meioSessions') || '[]');
      const docInfo = existingSessions.find(doc => doc.session_id === documentId || doc.id === documentId);
      
      setCurrentDocumentData({
        id: documentId,
        ...response.data,
        columns: Object.keys(response.data.data[0] || {}),
        row_count: response.data.data.length,
        detected_text_column: docInfo?.detected_text_column
      });
      setDetectedTextColumn(docInfo?.detected_text_column || '');
      setSelectedTextColumn(docInfo?.detected_text_column || '');
      setActiveNav('data-viewer');
    } catch (error) {
      console.error('ViewDocument error:', error);
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
      const requestPayload = {
        session_id: selectedDocument
      };
      
      // Include selected text column if user has overridden the detection
      if (selectedTextColumn && selectedTextColumn !== detectedTextColumn) {
        requestPayload.text_column_override = selectedTextColumn;
      }
      
      const response = await axios.post(`${FINAL_API_BASE}/analyze`, requestPayload);
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
          session_id: selectedDocument,
          text_column_override: selectedTextColumn !== detectedTextColumn ? selectedTextColumn : undefined
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
                  // Ensure exposure statistics are included from streaming response
                  const completeData = {
                    ...data,
                    exposure_statistics: data.exposure_statistics || {
                      max_exposure: 0,
                      min_exposure: 0,
                      avg_exposure: 0,
                      total_exposure: 0
                    },
                    profile_distribution: data.profile_distribution || {}
                  };
                  setAnalysisResults(completeData);
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
      const requestPayload = {
        session_id: selectedDocument
      };
      
      // Include selected text column if user has overridden the detection
      if (selectedTextColumn && selectedTextColumn !== detectedTextColumn) {
        requestPayload.text_column_override = selectedTextColumn;
      }
      
      const response = await axios.post(`${FINAL_API_BASE}/mitigate`, requestPayload);
      
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
          session_id: selectedDocument,
          text_column_override: selectedTextColumn !== detectedTextColumn ? selectedTextColumn : undefined
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

  const analyzeInfluentialVoices = async () => {
    if (!selectedDocument) {
      showMessage('Please select a document to analyze', 'error');
      return;
    }

    setLoading(true);
    try {
      const requestPayload = {
        session_id: selectedDocument,
        exposure_threshold: exposureThreshold,
        top_count: topCount,
        use_threshold: useThreshold
      };
      
      // Include selected text column if user has overridden the detection
      if (selectedTextColumn && selectedTextColumn !== detectedTextColumn) {
        requestPayload.text_column_override = selectedTextColumn;
      }
      
      const response = await axios.post(`${FINAL_API_BASE}/analyze-influential`, requestPayload);
      setInfluentialVoices(response.data);
      showMessage('Influential voices analysis completed successfully!');
    } catch (error) {
      showMessage('Error analyzing influential voices: ' + (error.response?.data?.detail || error.message), 'error');
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
      case 'influential-voices': return 'Influential Voices Analysis';
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
            <img
              src="/assets/MCMC_Logo.png"
              alt="MCMC Logo"
              style={{
                height: '60px',
                width: 'auto',
                objectFit: 'contain',
                marginBottom: '1rem'
              }}
            />
            <LoginTitle>MCMC Sentiment Analysis</LoginTitle>
            <LoginSubtitle>Malaysian Communications and Multimedia Commission</LoginSubtitle>
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
              Sign In
            </Button>
            
            <div style={{marginTop: '1rem', fontSize: '0.75rem', color: '#6b7280', textAlign: 'center'}}>
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
          <img
            src="/assets/MCMC_Logo.png"
            alt="MCMC Logo"
            style={{
              height: '40px',
              width: 'auto',
              objectFit: 'contain'
            }}
          />
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
            $active={activeNav === 'influential-voices'}
            $collapsed={sidebarCollapsed}
            onClick={() => setActiveNav('influential-voices')}
          >
            <Target size={20} />
            <span>Influential Voices</span>
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

              <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem'}}>
                <Card>
                  <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>Recent Documents</h2>
                  {documents.length === 0 ? (
                    <p style={{color: '#6b7280', textAlign: 'center', padding: '2rem'}}>
                      No documents uploaded yet. Start by uploading a document.
                    </p>
                  ) : (
                    documents.slice(0, 5).map((doc, index) => (
                      <DocumentCard key={`dashboard-doc-${doc.id || doc.session_id || index}`}>
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
                            <Button $variant="secondary" onClick={() => viewDocument(doc.session_id || doc.id)}>
                              <Eye size={16} />
                              View
                            </Button>
                          </div>
                        </DocumentHeader>
                      </DocumentCard>
                    ))
                  )}
                </Card>

                <Card>
                  <h2 style={{marginBottom: '1.5rem', color: '#1f2937', display: 'flex', alignItems: 'center'}}>
                    <Database size={20} style={{marginRight: '0.5rem'}} />
                    Saved Analyses
                  </h2>
                  
                  <FormGroup>
                    <Label>Quick Load Analysis</Label>
                    <div style={{display: 'flex', gap: '0.5rem'}}>
                      <Select
                        value={selectedSavedAnalysis}
                        onChange={(e) => setSelectedSavedAnalysis(e.target.value)}
                        style={{flex: 1}}
                      >
                        <option value="">Choose saved analysis...</option>
                        {savedAnalyses.map((analysis, index) => (
                          <option key={`saved-analysis-${analysis.id || index}`} value={analysis.id}>
                            {analysis.analysis_name} ({analysis.analysis_type})
                          </option>
                        ))}
                      </Select>
                      <Button
                        $variant="secondary"
                        onClick={() => {
                          if (selectedSavedAnalysis) {
                            loadSavedAnalysis(selectedSavedAnalysis);
                          }
                        }}
                        disabled={!selectedSavedAnalysis}
                      >
                        <RefreshCw size={16} />
                        Load
                      </Button>
                    </div>
                  </FormGroup>

                  {savedAnalyses.length === 0 ? (
                    <p style={{color: '#6b7280', textAlign: 'center', padding: '2rem'}}>
                      No saved analyses yet. Complete an analysis to save it.
                    </p>
                  ) : (
                    <div style={{maxHeight: '300px', overflowY: 'auto'}}>
                      {savedAnalyses.slice(0, 5).map((analysis, index) => (
                        <DocumentCard key={`saved-analysis-card-${analysis.id || index}`} style={{marginBottom: '1rem'}}>
                          <DocumentHeader>
                            <div>
                              <DocumentTitle style={{fontSize: '1rem'}}>{analysis.analysis_name}</DocumentTitle>
                              <DocumentMeta>
                                Type: {analysis.analysis_type} |
                                Created: {new Date(analysis.created_date).toLocaleDateString()}
                              </DocumentMeta>
                            </div>
                            <div style={{display: 'flex', gap: '0.5rem'}}>
                              <Button
                                $variant="secondary"
                                style={{padding: '0.5rem'}}
                                onClick={() => loadSavedAnalysis(analysis.id)}
                              >
                                <RefreshCw size={14} />
                              </Button>
                              <Button
                                $variant="danger"
                                style={{padding: '0.5rem'}}
                                onClick={() => deleteSavedAnalysis(analysis.id)}
                              >
                                <Trash2 size={14} />
                              </Button>
                            </div>
                          </DocumentHeader>
                        </DocumentCard>
                      ))}
                    </div>
                  )}
                </Card>
              </div>

              {/* Dashboard Visualizations for Loaded Analysis */}
              {(analysisResults || influentialVoices || responses) && (
                <div style={{marginTop: '2rem'}}>
                  <h2 style={{marginBottom: '1.5rem', color: '#1f2937', display: 'flex', alignItems: 'center'}}>
                    <BarChart3 size={20} style={{marginRight: '0.5rem'}} />
                    Analysis Visualizations
                  </h2>
                  
                  <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem'}}>
                    {/* Sentiment Distribution Chart */}
                    {analysisResults && (
                      <ChartContainer>
                        <ChartTitle>Sentiment Distribution</ChartTitle>
                        <PieChartContainer>
                          <PieChart colors={`
                            #10b981 0deg ${(analysisResults.statistics?.positive || 0) / (analysisResults.analyzed_data?.length || 1) * 360}deg,
                            #ef4444 ${(analysisResults.statistics?.positive || 0) / (analysisResults.analyzed_data?.length || 1) * 360}deg ${((analysisResults.statistics?.positive || 0) + (analysisResults.statistics?.negative || 0)) / (analysisResults.analyzed_data?.length || 1) * 360}deg,
                            #6b7280 ${((analysisResults.statistics?.positive || 0) + (analysisResults.statistics?.negative || 0)) / (analysisResults.analyzed_data?.length || 1) * 360}deg 360deg
                          `}>
                            <PieCenter>
                              {analysisResults.analyzed_data?.length || 0}
                            </PieCenter>
                          </PieChart>
                          <LegendContainer>
                            <LegendItem>
                              <LegendColor color="#10b981" />
                              Positive: {analysisResults.statistics?.positive || 0}
                            </LegendItem>
                            <LegendItem>
                              <LegendColor color="#ef4444" />
                              Negative: {analysisResults.statistics?.negative || 0}
                            </LegendItem>
                            <LegendItem>
                              <LegendColor color="#6b7280" />
                              Neutral: {analysisResults.statistics?.neutral || 0}
                            </LegendItem>
                          </LegendContainer>
                        </PieChartContainer>
                      </ChartContainer>
                    )}

                    {/* Exposure Score Distribution */}
                    {analysisResults?.exposure_statistics && (
                      <ChartContainer>
                        <ChartTitle>Exposure Score Analysis</ChartTitle>
                        <BarChart>
                          <BarItem>
                            <BarLabel>Max Exposure</BarLabel>
                            <BarContainer>
                              <BarFill
                                color="#f59e0b"
                                percentage={100}
                              />
                            </BarContainer>
                            <BarValue>{analysisResults.exposure_statistics.max_exposure?.toLocaleString()}</BarValue>
                          </BarItem>
                          <BarItem>
                            <BarLabel>Avg Exposure</BarLabel>
                            <BarContainer>
                              <BarFill
                                color="#8b5cf6"
                                percentage={(analysisResults.exposure_statistics.avg_exposure / analysisResults.exposure_statistics.max_exposure) * 100}
                              />
                            </BarContainer>
                            <BarValue>{analysisResults.exposure_statistics.avg_exposure?.toLocaleString()}</BarValue>
                          </BarItem>
                          <BarItem>
                            <BarLabel>Total Exposure</BarLabel>
                            <BarContainer>
                              <BarFill
                                color="#2596be"
                                percentage={80}
                              />
                            </BarContainer>
                            <BarValue>{analysisResults.exposure_statistics.total_exposure?.toLocaleString()}</BarValue>
                          </BarItem>
                        </BarChart>
                      </ChartContainer>
                    )}

                    {/* Profile Distribution Chart */}
                    {analysisResults?.profile_distribution && (
                      <ChartContainer>
                        <ChartTitle>Profile Type Distribution</ChartTitle>
                        <BarChart>
                          {Object.entries(analysisResults.profile_distribution).map(([profile, count], index) => {
                            const colors = ['#10b981', '#ef4444', '#f59e0b', '#8b5cf6'];
                            const maxCount = Math.max(...Object.values(analysisResults.profile_distribution));
                            return (
                              <BarItem key={profile}>
                                <BarLabel>{profile}</BarLabel>
                                <BarContainer>
                                  <BarFill
                                    color={colors[index % colors.length]}
                                    percentage={(count / maxCount) * 100}
                                  />
                                </BarContainer>
                                <BarValue>{count}</BarValue>
                              </BarItem>
                            );
                          })}
                        </BarChart>
                      </ChartContainer>
                    )}

                    {/* Influential Voices Summary */}
                    {influentialVoices && (
                      <ChartContainer>
                        <ChartTitle>Influential Voices Summary</ChartTitle>
                        <div style={{display: 'flex', flexDirection: 'column', gap: '1rem'}}>
                          <div style={{display: 'flex', justifyContent: 'space-between'}}>
                            <span style={{color: '#6b7280'}}>Priority Voices:</span>
                            <strong>{influentialVoices.statistics?.priority_voices || 0}</strong>
                          </div>
                          <div style={{display: 'flex', justifyContent: 'space-between'}}>
                            <span style={{color: '#6b7280'}}>Negative High-Impact:</span>
                            <strong style={{color: '#ef4444'}}>{influentialVoices.statistics?.negative_priority_voices || 0}</strong>
                          </div>
                          <div style={{display: 'flex', justifyContent: 'space-between'}}>
                            <span style={{color: '#6b7280'}}>Max Exposure:</span>
                            <strong>{influentialVoices.statistics?.exposure_stats?.max_exposure?.toLocaleString() || '0'}</strong>
                          </div>
                          <div style={{display: 'flex', justifyContent: 'space-between'}}>
                            <span style={{color: '#6b7280'}}>Counter-Statements:</span>
                            <strong style={{color: '#10b981'}}>{influentialVoices.priority_voices?.filter(v => v.counter_statement && v.counter_statement !== '').length || 0}</strong>
                          </div>
                        </div>
                      </ChartContainer>
                    )}
                  </div>
                </div>
              )}
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
                  Or click to select file
                </p>
              </DropZone>

              {(loading || uploadProgress > 0) && (
                <ProgressContainer>
                  <ProgressText>{uploadStatus}</ProgressText>
                  <ProgressBar>
                    <ProgressFill $progress={uploadProgress} />
                  </ProgressBar>
                  <ProgressText>{uploadProgress}%</ProgressText>
                </ProgressContainer>
              )}

              <div style={{marginTop: '2rem'}}>
                <h3 style={{marginBottom: '1rem', color: '#1f2937'}}>Upload History</h3>
                {documents.length === 0 ? (
                  <p style={{color: '#6b7280', textAlign: 'center', padding: '2rem'}}>
                    No documents uploaded yet.
                  </p>
                ) : (
                  documents.map((doc, index) => (
                    <DocumentCard key={`upload-doc-${doc.id || doc.session_id || index}`}>
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
                          <Button $variant="secondary" onClick={() => viewDocument(doc.session_id || doc.id)}>
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
                    documents.map((doc, index) => (
                      <DocumentCard key={`viewer-doc-${doc.id || doc.session_id || index}`}>
                        <DocumentHeader>
                          <div>
                            <DocumentTitle>{doc.filename}</DocumentTitle>
                            <DocumentMeta>
                              Uploaded: {new Date(doc.upload_date).toLocaleDateString()} |
                              Rows: {doc.row_count} |
                              Status: <StatusBadge $status={doc.status}>{doc.status}</StatusBadge>
                            </DocumentMeta>
                          </div>
                          <Button onClick={() => viewDocument(doc.session_id || doc.id)}>
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
                      Uploaded: {currentDocumentData.upload_date ? new Date(currentDocumentData.upload_date).toLocaleDateString() : 'Unknown'} |
                      Rows: {currentDocumentData.row_count || 0} |
                      Columns: {currentDocumentData.columns?.length || 0}
                    </DocumentMeta>
                  </Card>

                  <FastDataTable
                    data={currentDocumentData.data?.map(row =>
                      currentDocumentData.columns?.map(col => row[col] || '')
                    )}
                    columns={currentDocumentData.columns}
                    title={`Document Data (${currentDocumentData.row_count} rows)`}
                  />
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
                  onChange={(e) => {
                    const docId = e.target.value;
                    setSelectedDocument(docId);
                    
                    // Update text column selection when document changes
                    if (docId) {
                      const existingSessions = JSON.parse(localStorage.getItem('meioSessions') || '[]');
                      const docInfo = existingSessions.find(doc => doc.session_id === docId || doc.id === docId);
                      const detectedCol = docInfo?.detected_text_column || '';
                      setDetectedTextColumn(detectedCol);
                      setSelectedTextColumn(detectedCol);
                    }
                  }}
                >
                  <option value="">Choose a document...</option>
                  {documents.map((doc, index) => (
                    <option key={`analysis-option-${doc.id || doc.session_id || index}`} value={doc.session_id || doc.id}>
                      {doc.filename} ({doc.row_count} rows)
                    </option>
                  ))}
                </Select>
              </FormGroup>

              {selectedDocument && currentDocumentData && (
                <FormGroup>
                  <Label>Text Column for Analysis</Label>
                  <Select
                    value={selectedTextColumn}
                    onChange={(e) => setSelectedTextColumn(e.target.value)}
                  >
                    {currentDocumentData.columns?.map(col => (
                      <option key={col} value={col}>
                        {col} {col === detectedTextColumn ? '(Auto-detected)' : ''}
                      </option>
                    ))}
                  </Select>
                  {detectedTextColumn && (
                    <div style={{marginTop: '0.5rem', fontSize: '0.875rem', color: '#6b7280'}}>
                      💡 Auto-detected text column: <strong>{detectedTextColumn}</strong>
                      {selectedTextColumn !== detectedTextColumn && (
                        <span style={{color: '#f59e0b'}}> (You've overridden this selection)</span>
                      )}
                    </div>
                  )}
                </FormGroup>
              )}

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
                              {item.topics.map((topic, idx) => (
                                <TopicBadge key={`streaming-topic-${item.index}-${idx}`}>
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
                    {analysisResults.exposure_statistics && (
                      <>
                        <StatCard>
                          <StatNumber color="#f59e0b">{analysisResults.exposure_statistics.max_exposure?.toLocaleString() || '0'}</StatNumber>
                          <StatLabel>Max Exposure Score</StatLabel>
                        </StatCard>
                        <StatCard>
                          <StatNumber color="#8b5cf6">{analysisResults.exposure_statistics.total_exposure?.toLocaleString() || '0'}</StatNumber>
                          <StatLabel>Total Exposure</StatLabel>
                        </StatCard>
                      </>
                    )}
                  </StatsGrid>

                  {/* Exposure Filter */}
                  <div style={{display: 'flex', gap: '1rem', marginBottom: '1.5rem', alignItems: 'center'}}>
                    <Label style={{margin: 0}}>Filter by Exposure Score:</Label>
                    <Select
                      value={filterSentiment}
                      onChange={(e) => setFilterSentiment(e.target.value)}
                      style={{width: 'auto', minWidth: '150px'}}
                    >
                      <option value="all">All Posts</option>
                      <option value="high_exposure">High Exposure (≥500K)</option>
                      <option value="medium_exposure">Medium Exposure (100K-500K)</option>
                      <option value="low_exposure">Low Exposure (&lt;100K)</option>
                    </Select>
                    <Button
                      $variant="secondary"
                      style={{padding: '0.5rem'}}
                      onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
                    >
                      {sortOrder === 'desc' ? <SortDesc size={16} /> : <SortAsc size={16} />}
                      Sort by Exposure {sortOrder === 'desc' ? 'High to Low' : 'Low to High'}
                    </Button>
                  </div>

                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem'}}>
                    <h3 style={{color: '#1f2937', margin: 0}}>Analysis Results</h3>
                    <Button
                      style={{background: '#10b981'}}
                      onClick={() => {
                        const analysisName = prompt('Enter a name for this analysis:', `Sentiment_Analysis_${new Date().toISOString().slice(0,10)}`);
                        if (analysisName) {
                          saveAnalysisSession(analysisName, 'sentiment', analysisResults);
                        }
                      }}
                    >
                      <Database size={16} />
                      Save Analysis
                    </Button>
                  </div>

                  <FastDataTable
                    data={analysisResults.analyzed_data?.filter(row => {
                      const exposureScore = row.exposure_score || 0;
                      if (filterSentiment === 'all') return true;
                      if (filterSentiment === 'high_exposure') return exposureScore >= 500000;
                      if (filterSentiment === 'medium_exposure') return exposureScore >= 100000 && exposureScore < 500000;
                      if (filterSentiment === 'low_exposure') return exposureScore < 100000;
                      return true;
                    })
                    .sort((a, b) => {
                      const aExp = a.exposure_score || 0;
                      const bExp = b.exposure_score || 0;
                      return sortOrder === 'desc' ? bExp - aExp : aExp - bExp;
                    })
                    .map(row => {
                      const textContent = Object.values(row).find(val =>
                        typeof val === 'string' && val.length > 10 &&
                        !['sentiment', 'confidence', 'topic', 'topics', 'exposure_score', 'profile_tag', 'author_url', 'content_url'].includes(val)
                      ) || 'No text content';
                      
                      const topics = Array.isArray(row.topics) ? row.topics.join(', ') : (row.topic || 'General');
                      const confidence = row.confidence && !isNaN(row.confidence) ? (row.confidence * 100).toFixed(1) : '0.0';
                      
                      // Use max_exp directly if available, otherwise use exposure_score
                      let exposureScore = 0;
                      if (row.max_exp !== undefined && row.max_exp !== null && row.max_exp !== '') {
                        try {
                          exposureScore = typeof row.max_exp === 'string' ?
                            parseFloat(row.max_exp.replace(',', '')) :
                            parseFloat(row.max_exp);
                        } catch (e) {
                          exposureScore = row.exposure_score || 0;
                        }
                      } else {
                        exposureScore = row.exposure_score || 0;
                      }
                      
                      const exposureScoreFormatted = exposureScore.toLocaleString();
                      const profileTag = row.profile_tag || 'UNTAGGED';
                      
                      return [
                        String(textContent),
                        row.sentiment || 'neutral',
                        confidence,
                        topics,
                        exposureScore,
                        profileTag,
                        row.author_url || '-',
                        row.content_url || '-'
                      ];
                    })}
                    columns={['Content', 'Sentiment', 'Confidence', 'Topics', 'Exposure Score', 'Profile Type', 'Author URL', 'Content URL']}
                    title={`Analysis Results (${analysisResults.analyzed_data?.filter(row => {
                      const exposureScore = row.exposure_score || 0;
                      if (filterSentiment === 'all') return true;
                      if (filterSentiment === 'high_exposure') return exposureScore >= 500000;
                      if (filterSentiment === 'medium_exposure') return exposureScore >= 100000 && exposureScore < 500000;
                      if (filterSentiment === 'low_exposure') return exposureScore < 100000;
                      return true;
                    }).length || 0} shown)`}
                  />
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
                  onChange={(e) => {
                    const docId = e.target.value;
                    setSelectedDocument(docId);
                    
                    // Update text column selection when document changes
                    if (docId) {
                      const existingSessions = JSON.parse(localStorage.getItem('meioSessions') || '[]');
                      const docInfo = existingSessions.find(doc => doc.session_id === docId || doc.id === docId);
                      const detectedCol = docInfo?.detected_text_column || '';
                      setDetectedTextColumn(detectedCol);
                      setSelectedTextColumn(detectedCol);
                    }
                  }}
                >
                  <option value="">Choose a document...</option>
                  {documents.map((doc, index) => (
                    <option key={`response-option-${doc.id || doc.session_id || index}`} value={doc.session_id || doc.id}>
                      {doc.filename} ({doc.row_count} rows)
                    </option>
                  ))}
                </Select>
              </FormGroup>

              {selectedDocument && currentDocumentData && (
                <FormGroup>
                  <Label>Text Column for Response Generation</Label>
                  <Select
                    value={selectedTextColumn}
                    onChange={(e) => setSelectedTextColumn(e.target.value)}
                  >
                    {currentDocumentData.columns?.map(col => (
                      <option key={col} value={col}>
                        {col} {col === detectedTextColumn ? '(Auto-detected)' : ''}
                      </option>
                    ))}
                  </Select>
                  {detectedTextColumn && (
                    <div style={{marginTop: '0.5rem', fontSize: '0.875rem', color: '#6b7280'}}>
                      💡 Using text column: <strong>{selectedTextColumn}</strong>
                      {selectedTextColumn !== detectedTextColumn && (
                        <span style={{color: '#f59e0b'}}> (Overridden from auto-detected: {detectedTextColumn})</span>
                      )}
                    </div>
                  )}
                </FormGroup>
              )}

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
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem'}}>
                    <h3 style={{color: '#1f2937', margin: 0}}>Generated Responses</h3>
                    <Button
                      style={{background: '#10b981'}}
                      onClick={() => {
                        const analysisName = prompt('Enter a name for this analysis:', `Response_Generation_${new Date().toISOString().slice(0,10)}`);
                        if (analysisName) {
                          saveAnalysisSession(analysisName, 'responses', responses);
                        }
                      }}
                    >
                      <Database size={16} />
                      Save Analysis
                    </Button>
                  </div>
                  
                  <FastDataTable
                    data={responses.responses?.map(response => [
                      response.original_text || '',
                      response.sentiment || 'neutral',
                      response.topics?.join(', ') || 'General',
                      response.generated_comment || ''
                    ])}
                    columns={['Original Content', 'Sentiment', 'Topics', 'Generated Response']}
                    title="Response Strategy"
                  />
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
                  {documents.filter(doc => doc.status === 'analyzed' || doc.status === 'completed').map((doc, index) => (
                    <option key={`report-option-${doc.id || doc.session_id || index}`} value={doc.session_id || doc.id}>
                      {doc.filename} ({doc.status})
                    </option>
                  ))}
                </Select>
              </FormGroup>

              <div style={{background: '#f8fafc', padding: '1.5rem', borderRadius: '8px', margin: '1.5rem 0'}}>
                <h3 style={{color: '#1f2937', marginBottom: '1rem'}}>Report Includes:</h3>
                <ul style={{color: '#6b7280', lineHeight: '1.6'}}>
                  <li>Executive Summary with key insights and recommendations</li>
                  <li>Volume Metrics and trend analysis</li>
                  <li>Detailed Sentiment Analysis breakdown</li>
                  <li>Audience Insights and demographics (where available)</li>
                  <li>Top Performing Content and engagement metrics</li>
                  <li>Key Topics & Hashtags analysis</li>
                  <li>Competitor & Industry Benchmarking</li>
                  <li>Crisis or Issue Tracking (if relevant)</li>
                  <li>Strategic Insights & Recommendations</li>
                  <li>Methodology & Data Sources</li>
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

          {/* Influential Voices Analysis */}
          {activeNav === 'influential-voices' && (
            <div>
              <Card>
                <h2 style={{marginBottom: '1.5rem', color: '#1f2937'}}>
                  <Target size={24} style={{marginRight: '0.5rem', verticalAlign: 'middle'}} />
                  Influential Voices Analysis
                </h2>
                
                <FormGroup>
                  <Label>Select Analyzed Document</Label>
                  <Select
                    value={selectedDocument}
                    onChange={(e) => {
                      const docId = e.target.value;
                      setSelectedDocument(docId);
                      
                      // Update text column selection when document changes
                      if (docId) {
                        const existingSessions = JSON.parse(localStorage.getItem('meioSessions') || '[]');
                        const docInfo = existingSessions.find(doc => doc.session_id === docId || doc.id === docId);
                        const detectedCol = docInfo?.detected_text_column || '';
                        setDetectedTextColumn(detectedCol);
                        setSelectedTextColumn(detectedCol);
                      }
                    }}
                  >
                    <option value="">Choose a document...</option>
                    {documents.filter(doc => doc.status === 'analyzed' || doc.status === 'completed').map((doc, index) => (
                      <option key={`influential-option-${doc.id || doc.session_id || index}`} value={doc.session_id || doc.id}>
                        {doc.filename} ({doc.row_count} rows)
                      </option>
                    ))}
                  </Select>
                </FormGroup>

                {selectedDocument && currentDocumentData && (
                  <FormGroup>
                    <Label>Text Column for Analysis</Label>
                    <Select
                      value={selectedTextColumn}
                      onChange={(e) => setSelectedTextColumn(e.target.value)}
                    >
                      {currentDocumentData.columns?.map(col => (
                        <option key={col} value={col}>
                          {col} {col === detectedTextColumn ? '(Auto-detected)' : ''}
                        </option>
                      ))}
                    </Select>
                  </FormGroup>
                )}

                <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', margin: '1.5rem 0'}}>
                  <FormGroup>
                    <Label>Analysis Method</Label>
                    <Select
                      value={useThreshold ? 'threshold' : 'top'}
                      onChange={(e) => setUseThreshold(e.target.value === 'threshold')}
                    >
                      <option value="threshold">Use Exposure Threshold</option>
                      <option value="top">Use Top Count</option>
                    </Select>
                  </FormGroup>

                  {useThreshold ? (
                    <FormGroup>
                      <Label>Exposure Threshold (minimum exposure score)</Label>
                      <Input
                        type="number"
                        value={exposureThreshold}
                        onChange={(e) => setExposureThreshold(Number(e.target.value))}
                        placeholder="500000"
                        min="0"
                        step="1000"
                      />
                      <div style={{fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem'}}>
                        Posts with exposure score ≥ {exposureThreshold.toLocaleString()} will be analyzed
                      </div>
                    </FormGroup>
                  ) : (
                    <FormGroup>
                      <Label>Top Count (number of most influential)</Label>
                      <Input
                        type="number"
                        value={topCount}
                        onChange={(e) => setTopCount(Number(e.target.value))}
                        placeholder="5"
                        min="1"
                        max="50"
                      />
                      <div style={{fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem'}}>
                        Top {topCount} most influential voices will be analyzed
                      </div>
                    </FormGroup>
                  )}
                </div>

                <div style={{background: '#f0f9ff', padding: '1rem', borderRadius: '8px', margin: '1.5rem 0', border: '1px solid #bfdbfe'}}>
                  <h4 style={{color: '#1e40af', marginBottom: '0.5rem', display: 'flex', alignItems: 'center'}}>
                    <Crown size={16} style={{marginRight: '0.5rem'}} />
                    About Influential Voices Analysis
                  </h4>
                  <p style={{color: '#1e40af', fontSize: '0.875rem', lineHeight: '1.4', margin: 0}}>
                    This analysis identifies high-impact negative voices based on exposure scores (like Cyabra data) and generates strategic counter-statements.
                    Only negative sentiment posts from influential profiles will receive AI-generated responses.
                  </p>
                </div>

                <ButtonGroup>
                  <Button onClick={analyzeInfluentialVoices} disabled={!selectedDocument || loading}>
                    <Zap size={16} />
                    Analyze Influential Voices
                  </Button>
                  {influentialVoices && (
                    <Button
                      style={{background: '#10b981'}}
                      onClick={() => {
                        const analysisName = prompt('Enter a name for this analysis:', `Influential_Voices_${new Date().toISOString().slice(0,10)}`);
                        if (analysisName) {
                          saveAnalysisSession(analysisName, 'influential_voices', influentialVoices);
                        }
                      }}
                    >
                      <Database size={16} />
                      Save Analysis
                    </Button>
                  )}
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
              </Card>

              {/* Analysis Results */}
              {influentialVoices && (
                <Card>
                  <h3 style={{marginBottom: '1.5rem', color: '#1f2937', display: 'flex', alignItems: 'center'}}>
                    <Crown size={20} style={{marginRight: '0.5rem'}} />
                    Influential Voices Analysis Results
                  </h3>
                  
                  {/* Statistics */}
                  <StatsGrid>
                    <StatCard>
                      <StatNumber color="#2596be">{influentialVoices.statistics?.total_voices || 0}</StatNumber>
                      <StatLabel>Total Voices</StatLabel>
                    </StatCard>
                    <StatCard>
                      <StatNumber color="#ef4444">{influentialVoices.statistics?.priority_voices || 0}</StatNumber>
                      <StatLabel>Priority Voices</StatLabel>
                    </StatCard>
                    <StatCard>
                      <StatNumber color="#f59e0b">{influentialVoices.statistics?.negative_priority_voices || 0}</StatNumber>
                      <StatLabel>Negative High-Impact</StatLabel>
                    </StatCard>
                    <StatCard>
                      <StatNumber>{influentialVoices.statistics?.exposure_stats?.max_exposure?.toLocaleString() || '0'}</StatNumber>
                      <StatLabel>Max Exposure Score</StatLabel>
                    </StatCard>
                  </StatsGrid>

                  {/* Filters and Sorting */}
                  <div style={{display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap', alignItems: 'center'}}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                      <Filter size={16} />
                      <Label style={{margin: 0}}>Filter by Sentiment:</Label>
                      <Select
                        value={filterSentiment}
                        onChange={(e) => setFilterSentiment(e.target.value)}
                        style={{width: 'auto', minWidth: '120px'}}
                      >
                        <option value="all">All</option>
                        <option value="negative">Negative Only</option>
                        <option value="positive">Positive Only</option>
                        <option value="neutral">Neutral Only</option>
                      </Select>
                    </div>

                    <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                      {sortOrder === 'desc' ? <SortDesc size={16} /> : <SortAsc size={16} />}
                      <Label style={{margin: 0}}>Sort by:</Label>
                      <Select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value)}
                        style={{width: 'auto', minWidth: '140px'}}
                      >
                        <option value="exposure">Exposure Score</option>
                        <option value="confidence">Confidence</option>
                        <option value="influence_rank">Influence Rank</option>
                      </Select>
                      <Button
                        $variant="secondary"
                        style={{padding: '0.5rem'}}
                        onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
                      >
                        {sortOrder === 'desc' ? <SortDesc size={16} /> : <SortAsc size={16} />}
                      </Button>
                    </div>
                  </div>

                  {/* Influential Voices Table */}
                  {influentialVoices.priority_voices && influentialVoices.priority_voices.length > 0 && (
                    <FastDataTable
                      data={influentialVoices.priority_voices
                        .filter(voice => {
                          if (filterSentiment === 'all') return true;
                          return voice.sentiment === filterSentiment;
                        })
                        .sort((a, b) => {
                          let aVal = a[sortBy];
                          let bVal = b[sortBy];
                          
                          if (typeof aVal === 'string') {
                            aVal = aVal.toLowerCase();
                            bVal = bVal.toLowerCase();
                          }
                          
                          if (sortOrder === 'desc') {
                            return bVal > aVal ? 1 : -1;
                          } else {
                            return aVal > bVal ? 1 : -1;
                          }
                        })
                        .map(voice => [
                          voice.influence_rank || '-',
                          String(voice.original_text).substring(0, 150) + (voice.original_text.length > 150 ? '...' : ''),
                          voice.sentiment || 'neutral',
                          ((voice.confidence || 0) * 100).toFixed(1) + '%',
                          (voice.exposure_score || 0).toLocaleString(),
                          voice.profile_tag || 'UNTAGGED',
                          Array.isArray(voice.topics) ? voice.topics.join(', ') : (voice.topics || 'General'),
                          voice.counter_statement || (voice.sentiment === 'negative' ? 'No counter-statement generated' : 'N/A (non-negative)'),
                          voice.author_url || '-',
                          voice.content_url || '-'
                        ])}
                      columns={[
                        'Rank', 'Content', 'Sentiment', 'Confidence', 'Exposure Score',
                        'Profile Type', 'Topics', 'Counter Statement', 'Author URL', 'Content URL'
                      ]}
                      title={`Influential Voices (${influentialVoices.priority_voices.filter(voice => {
                        if (filterSentiment === 'all') return true;
                        return voice.sentiment === filterSentiment;
                      }).length} shown)`}
                    />
                  )}

                  {/* Configuration Summary */}
                  <div style={{marginTop: '1.5rem', padding: '1rem', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e5e7eb'}}>
                    <h4 style={{color: '#374151', marginBottom: '0.5rem'}}>Analysis Configuration</h4>
                    <div style={{fontSize: '0.875rem', color: '#6b7280'}}>
                      <strong>Method:</strong> {influentialVoices.analysis_config?.use_threshold ? 'Exposure Threshold' : 'Top Count'} |
                      <strong> Threshold:</strong> {influentialVoices.analysis_config?.exposure_threshold?.toLocaleString() || 'N/A'} |
                      <strong> Text Column:</strong> {influentialVoices.analysis_config?.text_column || 'Auto-detected'} |
                      <strong> Detected Columns:</strong> {Object.keys(influentialVoices.analysis_config?.detected_columns || {}).length} Cyabra columns found
                    </div>
                  </div>
                </Card>
              )}
            </div>
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