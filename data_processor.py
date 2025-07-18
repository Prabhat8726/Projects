import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any

class DataProcessor:
    def __init__(self):
        self.processed_data = None
        self.data_stats = {}
    
    def process_csv_data(self, df: pd.DataFrame, question_col: str, response_col: str) -> Dict[str, List[str]]:
        """Process CSV data for model training"""
        try:
            # Validate columns exist
            if question_col not in df.columns:
                raise ValueError(f"Question column '{question_col}' not found in dataset")
            if response_col not in df.columns:
                raise ValueError(f"Response column '{response_col}' not found in dataset")
            
            # Clean the data
            df_clean = self.clean_data(df[[question_col, response_col]])
            
            # Extract questions and responses
            questions = df_clean[question_col].tolist()
            responses = df_clean[response_col].tolist()
            
            # Validate data quality
            self.validate_data_quality(questions, responses)
            
            # Store processed data
            self.processed_data = {
                'questions': questions,
                'responses': responses
            }
            
            # Calculate statistics
            self.calculate_data_stats(questions, responses)
            
            return self.processed_data
            
        except Exception as e:
            raise Exception(f"Data processing failed: {str(e)}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataframe"""
        # Remove rows with missing values
        df_clean = df.dropna()
        
        # Remove duplicate questions
        df_clean = df_clean.drop_duplicates(subset=[df_clean.columns[0]])
        
        # Clean text data
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].str.strip()
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
        
        # Remove empty strings
        df_clean = df_clean[(df_clean.iloc[:, 0] != '') & (df_clean.iloc[:, 1] != '')]
        
        # Filter out very short questions/responses
        df_clean = df_clean[
            (df_clean.iloc[:, 0].str.len() >= 10) & 
            (df_clean.iloc[:, 1].str.len() >= 20)
        ]
        
        return df_clean
    
    def validate_data_quality(self, questions: List[str], responses: List[str]) -> None:
        """Validate the quality of processed data"""
        if len(questions) != len(responses):
            raise ValueError("Number of questions and responses must match")
        
        if len(questions) == 0:
            raise ValueError("No valid data found after cleaning")
        
        if len(questions) < 10:
            raise ValueError("Dataset too small. At least 10 question-response pairs required")
        
        # Check for diversity in responses
        unique_responses = len(set(responses))
        if unique_responses < 2:
            raise ValueError("Dataset needs more diverse responses for training")
        
        # Check average lengths
        avg_question_length = np.mean([len(q.split()) for q in questions])
        avg_response_length = np.mean([len(r.split()) for r in responses])
        
        if avg_question_length < 3:
            raise ValueError("Questions appear to be too short on average")
        
        if avg_response_length < 5:
            raise ValueError("Responses appear to be too short on average")
    
    def calculate_data_stats(self, questions: List[str], responses: List[str]) -> None:
        """Calculate statistics about the processed data"""
        self.data_stats = {
            'total_samples': len(questions),
            'unique_responses': len(set(responses)),
            'avg_question_length': np.mean([len(q.split()) for q in questions]),
            'avg_response_length': np.mean([len(r.split()) for r in responses]),
            'min_question_length': min([len(q.split()) for q in questions]),
            'max_question_length': max([len(q.split()) for q in questions]),
            'min_response_length': min([len(r.split()) for r in responses]),
            'max_response_length': max([len(r.split()) for r in responses]),
            'response_distribution': self.get_response_distribution(responses)
        }
    
    def get_response_distribution(self, responses: List[str]) -> Dict[str, int]:
        """Get distribution of responses"""
        response_counts = {}
        for response in responses:
            response_counts[response] = response_counts.get(response, 0) + 1
        
        # Sort by frequency
        sorted_responses = sorted(response_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_responses[:10])  # Return top 10 most frequent responses
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get data statistics"""
        return self.data_stats
    
    def export_processed_data(self, filepath: str) -> None:
        """Export processed data to CSV"""
        if self.processed_data is None:
            raise ValueError("No processed data available to export")
        
        df = pd.DataFrame(self.processed_data)
        df.to_csv(filepath, index=False)
    
    def detect_data_issues(self) -> List[str]:
        """Detect potential issues in the data"""
        issues = []
        
        if not self.processed_data:
            return ["No processed data available"]
        
        questions = self.processed_data['questions']
        responses = self.processed_data['responses']
        
        # Check for imbalanced data
        response_counts = {}
        for response in responses:
            response_counts[response] = response_counts.get(response, 0) + 1
        
        max_count = max(response_counts.values())
        min_count = min(response_counts.values())
        
        if max_count / min_count > 10:
            issues.append("Highly imbalanced response distribution detected")
        
        # Check for very short content
        short_questions = sum(1 for q in questions if len(q.split()) < 5)
        if short_questions > len(questions) * 0.2:
            issues.append("More than 20% of questions are very short")
        
        short_responses = sum(1 for r in responses if len(r.split()) < 8)
        if short_responses > len(responses) * 0.2:
            issues.append("More than 20% of responses are very short")
        
        # Check for duplicate-like questions
        question_words = [set(q.lower().split()) for q in questions]
        similar_pairs = 0
        for i in range(len(question_words)):
            for j in range(i + 1, len(question_words)):
                if len(question_words[i] & question_words[j]) / len(question_words[i] | question_words[j]) > 0.8:
                    similar_pairs += 1
        
        if similar_pairs > len(questions) * 0.1:
            issues.append("Many questions appear to be very similar")
        
        return issues
    
    def suggest_improvements(self) -> List[str]:
        """Suggest improvements for the dataset"""
        suggestions = []
        
        if not self.processed_data:
            return ["Process data first to get suggestions"]
        
        stats = self.data_stats
        
        if stats['total_samples'] < 100:
            suggestions.append("Consider adding more training samples for better model performance")
        
        if stats['unique_responses'] < 10:
            suggestions.append("Add more diverse responses to improve model capability")
        
        if stats['avg_question_length'] < 5:
            suggestions.append("Questions could be more detailed for better context")
        
        if stats['avg_response_length'] < 10:
            suggestions.append("Responses could be more comprehensive and detailed")
        
        # Check response distribution
        response_dist = stats['response_distribution']
        if response_dist:
            most_common_count = list(response_dist.values())[0]
            if most_common_count > stats['total_samples'] * 0.3:
                suggestions.append("Consider balancing response distribution - some responses are overrepresented")
        
        return suggestions
