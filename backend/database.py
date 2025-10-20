import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

class DocumentDatabase:
    def __init__(self, db_path: str = "meio_documents.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                row_count INTEGER,
                columns TEXT,
                status TEXT DEFAULT 'uploaded',
                metadata TEXT
            )
        ''')
        
        # Document data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                row_index INTEGER,
                row_data TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                row_index INTEGER,
                original_text TEXT,
                sentiment TEXT,
                confidence REAL,
                topics TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # Generated responses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generated_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                row_index INTEGER,
                original_text TEXT,
                sentiment TEXT,
                topics TEXT,
                generated_comment TEXT,
                generation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_document(self, filename: str, data: List[Dict], metadata: Dict = None) -> str:
        """Store a document and its data in the database"""
        document_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Extract columns from first row
            columns = list(data[0].keys()) if data else []
            
            # Insert document metadata
            cursor.execute('''
                INSERT INTO documents (id, filename, original_filename, file_size, row_count, columns, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                document_id,
                filename,
                filename,
                0,  # File size - can be calculated if needed
                len(data),
                json.dumps(columns),
                json.dumps(metadata or {})
            ))
            
            # Insert document data
            for idx, row in enumerate(data):
                cursor.execute('''
                    INSERT INTO document_data (document_id, row_index, row_data)
                    VALUES (?, ?, ?)
                ''', (document_id, idx, json.dumps(row)))
            
            conn.commit()
            return document_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_documents(self) -> List[Dict]:
        """Get list of all documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, original_filename, upload_date, row_count, status
            FROM documents
            ORDER BY upload_date DESC
        ''')
        
        documents = []
        for row in cursor.fetchall():
            documents.append({
                'id': row[0],
                'filename': row[1],
                'original_filename': row[2],
                'upload_date': row[3],
                'row_count': row[4],
                'status': row[5]
            })
        
        conn.close()
        return documents
    
    def get_document_data(self, document_id: str) -> Dict:
        """Get document data by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get document metadata
        cursor.execute('''
            SELECT filename, original_filename, upload_date, row_count, columns, metadata
            FROM documents WHERE id = ?
        ''', (document_id,))
        
        doc_result = cursor.fetchone()
        if not doc_result:
            conn.close()
            return None
        
        # Get document data
        cursor.execute('''
            SELECT row_index, row_data FROM document_data 
            WHERE document_id = ? ORDER BY row_index
        ''', (document_id,))
        
        data_rows = []
        for row in cursor.fetchall():
            data_rows.append(json.loads(row[1]))
        
        conn.close()
        
        return {
            'id': document_id,
            'filename': doc_result[0],
            'original_filename': doc_result[1],
            'upload_date': doc_result[2],
            'row_count': doc_result[3],
            'columns': json.loads(doc_result[4]) if doc_result[4] else [],
            'metadata': json.loads(doc_result[5]) if doc_result[5] else {},
            'data': data_rows
        }
    
    def store_analysis_results(self, document_id: str, results: List[Dict]):
        """Store sentiment analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Clear existing analysis results for this document
            cursor.execute('DELETE FROM analysis_results WHERE document_id = ?', (document_id,))
            
            # Insert new results
            for idx, result in enumerate(results):
                # Find the text content (first non-analysis field that's a string)
                text_content = ""
                for key, value in result.items():
                    if key not in ['sentiment', 'confidence', 'topic', 'topics'] and isinstance(value, str) and len(value) > 10:
                        text_content = value
                        break
                
                topics = result.get('topics', result.get('topic', ''))
                if isinstance(topics, list):
                    topics = ', '.join(topics)
                
                cursor.execute('''
                    INSERT INTO analysis_results (document_id, row_index, original_text, sentiment, confidence, topics)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    document_id,
                    idx,
                    text_content,
                    result.get('sentiment', 'neutral'),
                    result.get('confidence', 0.5),
                    topics
                ))
            
            # Update document status
            cursor.execute('UPDATE documents SET status = ? WHERE id = ?', ('analyzed', document_id))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_analysis_results(self, document_id: str) -> List[Dict]:
        """Get analysis results for a document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT row_index, original_text, sentiment, confidence, topics, analysis_date
            FROM analysis_results 
            WHERE document_id = ? 
            ORDER BY row_index
        ''', (document_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'row_index': row[0],
                'original_text': row[1],
                'sentiment': row[2],
                'confidence': row[3],
                'topics': row[4].split(', ') if row[4] else [],
                'analysis_date': row[5]
            })
        
        conn.close()
        return results
    
    def store_generated_responses(self, document_id: str, responses: List[Dict]):
        """Store generated responses"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Clear existing responses for this document
            cursor.execute('DELETE FROM generated_responses WHERE document_id = ?', (document_id,))
            
            # Insert new responses
            for idx, response in enumerate(responses):
                # Find the text content
                text_content = ""
                for key, value in response.items():
                    if key not in ['sentiment', 'confidence', 'topic', 'topics', 'generated_comment'] and isinstance(value, str) and len(value) > 10:
                        text_content = value
                        break
                
                topics = response.get('topics', response.get('topic', ''))
                if isinstance(topics, list):
                    topics = ', '.join(topics)
                
                cursor.execute('''
                    INSERT INTO generated_responses (document_id, row_index, original_text, sentiment, topics, generated_comment)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    document_id,
                    idx,
                    text_content,
                    response.get('sentiment', 'neutral'),
                    topics,
                    response.get('generated_comment', '')
                ))
            
            # Update document status
            cursor.execute('UPDATE documents SET status = ? WHERE id = ?', ('completed', document_id))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_generated_responses(self, document_id: str) -> List[Dict]:
        """Get generated responses for a document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT row_index, original_text, sentiment, topics, generated_comment, generation_date
            FROM generated_responses 
            WHERE document_id = ? 
            ORDER BY row_index
        ''', (document_id,))
        
        responses = []
        for row in cursor.fetchall():
            responses.append({
                'row_index': row[0],
                'original_text': row[1],
                'sentiment': row[2],
                'topics': row[3].split(', ') if row[3] else [],
                'generated_comment': row[4],
                'generation_date': row[5]
            })
        
        conn.close()
        return responses
    
    def delete_document(self, document_id: str):
        """Delete a document and all related data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM generated_responses WHERE document_id = ?', (document_id,))
            cursor.execute('DELETE FROM analysis_results WHERE document_id = ?', (document_id,))
            cursor.execute('DELETE FROM document_data WHERE document_id = ?', (document_id,))
            cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

# Global database instance
db = DocumentDatabase()