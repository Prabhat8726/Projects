import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from typing import Dict, List, Optional, Tuple
import logging

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.host = os.getenv('PGHOST')
        self.port = os.getenv('PGPORT')
        self.user = os.getenv('PGUSER')
        self.password = os.getenv('PGPASSWORD')
        self.database = os.getenv('PGDATABASE')
        
        # Initialize database tables
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return conn
        except Exception as e:
            logging.error(f"Database connection failed: {str(e)}")
            return None
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            with conn.cursor() as cursor:
                # Create health_data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS health_data (
                        id SERIAL PRIMARY KEY,
                        question TEXT NOT NULL,
                        response TEXT NOT NULL,
                        disease_category VARCHAR(100),
                        symptoms TEXT,
                        treatment TEXT,
                        severity VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        data_source VARCHAR(100)
                    )
                """)
                
                # Create models table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trained_models (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(100) NOT NULL,
                        accuracy FLOAT,
                        precision_score FLOAT,
                        recall_score FLOAT,
                        f1_score FLOAT,
                        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_metadata JSONB,
                        file_path VARCHAR(255)
                    )
                """)
                
                # Create chat_history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        user_question TEXT NOT NULL,
                        bot_response TEXT NOT NULL,
                        confidence_score FLOAT,
                        model_used VARCHAR(100),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        session_id VARCHAR(100)
                    )
                """)
                
                # Create dataset_sources table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dataset_sources (
                        id SERIAL PRIMARY KEY,
                        source_name VARCHAR(100) NOT NULL,
                        source_url TEXT,
                        description TEXT,
                        record_count INTEGER,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logging.info("Database tables initialized successfully")
                
        except Exception as e:
            logging.error(f"Database initialization failed: {str(e)}")
        finally:
            conn.close()
    
    def store_health_data(self, data: List[Dict]) -> bool:
        """Store health data in database"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            with conn.cursor() as cursor:
                for record in data:
                    cursor.execute("""
                        INSERT INTO health_data 
                        (question, response, disease_category, symptoms, treatment, severity, data_source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        record.get('question', ''),
                        record.get('response', ''),
                        record.get('disease_category', ''),
                        record.get('symptoms', ''),
                        record.get('treatment', ''),
                        record.get('severity', ''),
                        record.get('data_source', 'user_upload')
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logging.error(f"Failed to store health data: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_health_data(self, limit: int = 1000) -> List[Dict]:
        """Retrieve health data from database"""
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT question, response, disease_category, symptoms, treatment, severity
                    FROM health_data
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logging.error(f"Failed to retrieve health data: {str(e)}")
            return []
        finally:
            conn.close()
    
    def store_model_metadata(self, model_name: str, metrics: Dict, file_path: str) -> bool:
        """Store trained model metadata"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO trained_models 
                    (model_name, accuracy, precision_score, recall_score, f1_score, model_metadata, file_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    model_name,
                    metrics.get('accuracy', 0),
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('f1_score', 0),
                    json.dumps(metrics),
                    file_path
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logging.error(f"Failed to store model metadata: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_model_history(self) -> List[Dict]:
        """Get training history of models"""
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT model_name, accuracy, precision_score, recall_score, f1_score, 
                           training_date, file_path
                    FROM trained_models
                    ORDER BY training_date DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logging.error(f"Failed to retrieve model history: {str(e)}")
            return []
        finally:
            conn.close()
    
    def store_chat_interaction(self, user_question: str, bot_response: str, 
                              confidence: float, model_used: str, session_id: str) -> bool:
        """Store chat interaction"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO chat_history 
                    (user_question, bot_response, confidence_score, model_used, session_id)
                    VALUES (%s, %s, %s, %s, %s)
                """, (user_question, bot_response, confidence, model_used, session_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            logging.error(f"Failed to store chat interaction: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_chat_analytics(self) -> Dict:
        """Get chat analytics"""
        conn = self.get_connection()
        if not conn:
            return {}
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get total interactions
                cursor.execute("SELECT COUNT(*) as total_chats FROM chat_history")
                total_chats = cursor.fetchone()['total_chats']
                
                # Get average confidence
                cursor.execute("SELECT AVG(confidence_score) as avg_confidence FROM chat_history")
                avg_confidence = cursor.fetchone()['avg_confidence'] or 0
                
                # Get interactions by date
                cursor.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM chat_history
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                    LIMIT 7
                """)
                daily_stats = cursor.fetchall()
                
                return {
                    'total_chats': total_chats,
                    'avg_confidence': float(avg_confidence),
                    'daily_stats': [dict(row) for row in daily_stats]
                }
                
        except Exception as e:
            logging.error(f"Failed to get chat analytics: {str(e)}")
            return {}
        finally:
            conn.close()
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        conn = self.get_connection()
        if not conn:
            return {}
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT disease_category) as unique_diseases,
                        COUNT(DISTINCT data_source) as data_sources,
                        AVG(LENGTH(response)) as avg_response_length
                    FROM health_data
                """)
                
                stats = cursor.fetchone()
                return dict(stats) if stats else {}
                
        except Exception as e:
            logging.error(f"Failed to get dataset stats: {str(e)}")
            return {}
        finally:
            conn.close()