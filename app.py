import streamlit as st
import pandas as pd
import numpy as np
from model_trainer import HealthModelTrainer
from chat_handler import ChatHandler
from data_processor import DataProcessor
from utils import (display_metrics, show_medical_disclaimer, display_data_stats, 
                  display_data_issues, display_improvement_suggestions, 
                  create_sample_dataset, get_health_tips, validate_csv_structure)
from database_manager import DatabaseManager
from enhanced_datasets import EnhancedHealthDatasets
import os
import pickle
import time
import io
import uuid

# Page configuration
st.set_page_config(
    page_title="AI Health Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None
if 'chat_handler' not in st.session_state:
    st.session_state.chat_handler = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []
if 'show_sample_data' not in st.session_state:
    st.session_state.show_sample_data = False
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if 'enhanced_datasets' not in st.session_state:
    st.session_state.enhanced_datasets = EnhancedHealthDatasets()
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'use_enhanced_data' not in st.session_state:
    st.session_state.use_enhanced_data = False

def main():
    st.title("🏥 AI-Powered Virtual Health Chatbot")
    st.markdown("### Get instant health information powered by machine learning")
    
    # Medical disclaimer
    show_medical_disclaimer()
    
    # Add health tips in sidebar
    with st.sidebar:
        st.markdown("### 💡 Daily Health Tips")
        tips = get_health_tips()
        for tip in tips:
            st.info(tip)
    
    # Sidebar for model training
    with st.sidebar:
        st.header("📊 Model Training")
        
        # Dataset options
        dataset_option = st.selectbox(
            "Choose Dataset",
            ["Sample Dataset", "Enhanced Comprehensive Dataset", "Upload CSV"],
            help="Select the type of dataset to use for training"
        )
        
        # Option to use sample data
        if dataset_option == "Sample Dataset":
            if st.button("📋 Use Sample Dataset"):
                st.session_state.show_sample_data = True
                st.session_state.use_enhanced_data = False
                st.rerun()
        
        # Option to use enhanced comprehensive dataset
        elif dataset_option == "Enhanced Comprehensive Dataset":
            if st.button("🚀 Use Enhanced Dataset"):
                st.session_state.use_enhanced_data = True
                st.session_state.show_sample_data = False
                st.rerun()
            
            # Show info about enhanced dataset
            if st.session_state.use_enhanced_data:
                enhanced_df = st.session_state.enhanced_datasets.get_extended_dataset()
                st.info(f"Enhanced dataset contains {len(enhanced_df)} comprehensive health records with detailed responses")
                
                # Show dataset categories
                with st.expander("Dataset Categories"):
                    st.write("• Respiratory conditions")
                    st.write("• Cardiovascular health")
                    st.write("• Digestive issues")
                    st.write("• Mental health")
                    st.write("• Musculoskeletal conditions")
                    st.write("• Skin conditions")
                    st.write("• Neurological conditions")
                    st.write("• Endocrine disorders")
                    st.write("• Immune system")
                    st.write("• Pediatric health")
                    st.write("• Geriatric health")
                    st.write("• Preventive care")
                    
                if st.button("🚀 Train with Enhanced Data", type="primary"):
                    with st.spinner("Training model with enhanced comprehensive dataset..."):
                        try:
                            # Get enhanced dataset
                            enhanced_df = st.session_state.enhanced_datasets.get_extended_dataset()
                            
                            # Store in database
                            health_records = []
                            for _, row in enhanced_df.iterrows():
                                health_records.append({
                                    'question': row['question'],
                                    'response': row['response'],
                                    'data_source': 'enhanced_comprehensive_dataset'
                                })
                            
                            st.session_state.db_manager.store_health_data(health_records)
                            
                            # Process data
                            processed_data = st.session_state.data_processor.process_csv_data(
                                enhanced_df, 'question', 'response'
                            )
                            
                            # Initialize and train model
                            st.session_state.model_trainer = HealthModelTrainer(
                                test_size=0.2,
                                max_features=8000  # Increased for larger dataset
                            )
                            
                            # Train the model
                            training_results = st.session_state.model_trainer.train(processed_data)
                            
                            # Initialize chat handler
                            st.session_state.chat_handler = ChatHandler(
                                st.session_state.model_trainer
                            )
                            
                            st.session_state.model_trained = True
                            st.session_state.use_enhanced_data = False
                            
                            # Save model with timestamp
                            model_filename = f"enhanced_health_model_{int(time.time())}.pkl"
                            st.session_state.model_trainer.save_model(model_filename)
                            
                            # Store model metadata in database
                            st.session_state.db_manager.store_model_metadata(
                                model_filename, training_results, model_filename
                            )
                            
                            st.success(f"Enhanced model trained successfully! Using {len(enhanced_df)} health records.")
                            
                            # Display training results
                            display_metrics(training_results)
                            
                            # Get suggested questions
                            st.session_state.suggested_questions = st.session_state.chat_handler.suggest_questions()
                            
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
                            st.info("Please try again or contact support if the issue persists")
                            
        # CSV upload option
        elif dataset_option == "Upload CSV":
            st.session_state.show_sample_data = False
            st.session_state.use_enhanced_data = False
        
        # Show sample data option
        if st.session_state.show_sample_data:
            st.info("Using sample health dataset for demonstration")
            sample_df = create_sample_dataset()
            
            if st.button("🚀 Train with Sample Data", type="primary"):
                with st.spinner("Training model with sample data..."):
                    # Process sample data
                    processed_data = st.session_state.data_processor.process_csv_data(
                        sample_df, 'question', 'response'
                    )
                    
                    # Initialize and train model
                    st.session_state.model_trainer = HealthModelTrainer(
                        test_size=0.2,
                        max_features=5000
                    )
                    
                    # Train the model
                    training_results = st.session_state.model_trainer.train(processed_data)
                    
                    # Initialize chat handler
                    st.session_state.chat_handler = ChatHandler(
                        st.session_state.model_trainer
                    )
                    
                    st.session_state.model_trained = True
                    st.session_state.show_sample_data = False
                    st.success("Model trained successfully with sample data!")
                    
                    # Display training results
                    display_metrics(training_results)
                    
                    # Get suggested questions
                    st.session_state.suggested_questions = st.session_state.chat_handler.suggest_questions()
                    
                    st.rerun()
        
        # CSV file upload
        uploaded_file = st.file_uploader(
            "Upload Health Dataset (CSV)",
            type=['csv'],
            help="Upload a CSV file containing health questions and responses"
        )
        
        if uploaded_file is not None:
            try:
                # Load and preview data
                df = pd.read_csv(uploaded_file)
                st.success(f"Dataset loaded: {len(df)} rows")
                
                # Validate CSV structure
                validation_result = validate_csv_structure(df)
                
                if not validation_result['is_valid']:
                    st.error("CSV Validation Errors:")
                    for error in validation_result['errors']:
                        st.error(f"• {error}")
                    return
                
                # Show warnings if any
                if validation_result['warnings']:
                    st.warning("CSV Validation Warnings:")
                    for warning in validation_result['warnings']:
                        st.warning(f"• {warning}")
                
                # Show data preview
                with st.expander("Preview Dataset"):
                    st.dataframe(df.head())
                    
                    # Show column analysis
                    st.subheader("Column Analysis")
                    for col, info in validation_result['column_info'].items():
                        st.write(f"**{col}**: {info['non_null_count']} non-null values, {info['unique_count']} unique values")
                        if info['sample_values']:
                            st.write(f"Sample values: {info['sample_values']}")
                
                # Column selection
                columns = df.columns.tolist()
                question_col = st.selectbox(
                    "Select Question/Symptom Column",
                    columns,
                    help="Column containing health questions or symptoms"
                )
                response_col = st.selectbox(
                    "Select Response/Advice Column",
                    [col for col in columns if col != question_col],
                    help="Column containing health responses or advice"
                )
                
                # Show data quality analysis
                if st.button("🔍 Analyze Data Quality"):
                    with st.spinner("Analyzing data quality..."):
                        # Process data to get analysis
                        processed_data = st.session_state.data_processor.process_csv_data(
                            df, question_col, response_col
                        )
                        
                        # Display statistics
                        data_stats = st.session_state.data_processor.get_data_stats()
                        display_data_stats(data_stats)
                        
                        # Display issues and suggestions
                        issues = st.session_state.data_processor.detect_data_issues()
                        display_data_issues(issues)
                        
                        suggestions = st.session_state.data_processor.suggest_improvements()
                        display_improvement_suggestions(suggestions)
                
                # Training parameters
                st.subheader("Training Parameters")
                test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
                max_features = st.slider("Max Features (TF-IDF)", 1000, 10000, 5000, 500)
                
                # Advanced options
                with st.expander("Advanced Options"):
                    random_state = st.number_input("Random State", value=42, min_value=0)
                    ngram_range = st.selectbox(
                        "N-gram Range",
                        [(1, 1), (1, 2), (2, 2)],
                        index=1,
                        help="Range of n-grams to extract"
                    )
                
                # Train model button
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        try:
                            # Process data
                            processed_data = st.session_state.data_processor.process_csv_data(
                                df, question_col, response_col
                            )
                            
                            # Initialize and train model
                            st.session_state.model_trainer = HealthModelTrainer(
                                test_size=test_size,
                                max_features=max_features,
                                random_state=random_state
                            )
                            
                            # Train the model
                            training_results = st.session_state.model_trainer.train(processed_data)
                            
                            # Initialize chat handler
                            st.session_state.chat_handler = ChatHandler(
                                st.session_state.model_trainer
                            )
                            
                            st.session_state.model_trained = True
                            st.success("Model trained successfully!")
                            
                            # Display training results
                            display_metrics(training_results)
                            
                            # Save model
                            model_filename = f"health_model_{int(time.time())}.pkl"
                            st.session_state.model_trainer.save_model(model_filename)
                            st.info(f"Model saved as '{model_filename}'")
                            
                            # Get suggested questions
                            st.session_state.suggested_questions = st.session_state.chat_handler.suggest_questions()
                            
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
                            st.info("Please check your data format and try again")
                            
            except Exception as e:
                st.error(f"Error processing dataset: {str(e)}")
                st.info("Please ensure your CSV has proper columns for questions and responses")
        
        # Load existing model section
        st.subheader("Load Existing Model")
        model_files = [f for f in os.listdir('.') if f.startswith('health_model_') and f.endswith('.pkl')]
        
        if model_files:
            selected_model = st.selectbox("Select Model", model_files)
            if st.button("📁 Load Selected Model"):
                try:
                    st.session_state.model_trainer = HealthModelTrainer.load_model(selected_model)
                    st.session_state.chat_handler = ChatHandler(st.session_state.model_trainer)
                    st.session_state.model_trained = True
                    st.success(f"Model '{selected_model}' loaded successfully!")
                    
                    # Get suggested questions
                    st.session_state.suggested_questions = st.session_state.chat_handler.suggest_questions()
                    
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        else:
            st.info("No saved models found")
    
    # Main chat interface
    if st.session_state.model_trained:
        st.header("💬 Chat with Health Assistant")
        
        # Show suggested questions
        if st.session_state.suggested_questions:
            st.subheader("💡 Suggested Questions")
            cols = st.columns(len(st.session_state.suggested_questions))
            for i, question in enumerate(st.session_state.suggested_questions):
                with cols[i]:
                    if st.button(f"❓ {question}", key=f"suggestion_{i}"):
                        # Automatically ask the suggested question
                        st.session_state.current_question = question
                        st.rerun()
        
        # Handle suggested question
        if 'current_question' in st.session_state:
            user_input = st.session_state.current_question
            del st.session_state.current_question
            
            # Process the question
            with st.spinner("Processing your question..."):
                try:
                    response, confidence = st.session_state.chat_handler.get_response(user_input)
                    st.session_state.chat_history.append((user_input, response, confidence))
                    st.rerun()
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}"
                    st.session_state.chat_history.append((user_input, error_msg, None))
                    st.rerun()
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for i, (user_msg, bot_msg, confidence) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(user_msg)
                
                with st.chat_message("assistant"):
                    st.write(bot_msg)
                    if confidence is not None:
                        # Color-coded confidence
                        if confidence >= 0.8:
                            st.success(f"Confidence: {confidence:.2%}")
                        elif confidence >= 0.6:
                            st.warning(f"Confidence: {confidence:.2%}")
                        else:
                            st.error(f"Confidence: {confidence:.2%}")
        
        # Chat input
        user_input = st.chat_input("Ask me about health symptoms or conditions...")
        
        if user_input:
            # Add user message to history
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, confidence = st.session_state.chat_handler.get_response(user_input)
                        st.write(response)
                        
                        # Display confidence with color coding
                        if confidence >= 0.8:
                            st.success(f"Confidence: {confidence:.2%}")
                        elif confidence >= 0.6:
                            st.warning(f"Confidence: {confidence:.2%}")
                        else:
                            st.error(f"Confidence: {confidence:.2%}")
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_input, response, confidence))
                        
                        # Store in database
                        st.session_state.db_manager.store_chat_interaction(
                            user_input, response, confidence, 
                            "enhanced_health_model", st.session_state.session_id
                        )
                        
                    except Exception as e:
                        error_msg = f"I apologize, but I encountered an error processing your question: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append((user_input, error_msg, None))
        
        # Chat controls
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("📊 Analytics Dashboard"):
                # Get analytics from database
                analytics = st.session_state.db_manager.get_chat_analytics()
                dataset_stats = st.session_state.db_manager.get_dataset_stats()
                model_history = st.session_state.db_manager.get_model_history()
                
                if analytics:
                    st.subheader("📈 Chat Analytics")
                    
                    # Main metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Conversations", analytics.get('total_chats', 0))
                    with col_b:
                        st.metric("Average Confidence", f"{analytics.get('avg_confidence', 0):.2%}")
                    with col_c:
                        st.metric("Session Count", len(set([h[0] for h in st.session_state.chat_history])))
                    
                    # Daily stats
                    if analytics.get('daily_stats'):
                        st.subheader("Daily Activity")
                        daily_df = pd.DataFrame(analytics['daily_stats'])
                        st.bar_chart(daily_df.set_index('date'))
                
                if dataset_stats:
                    st.subheader("📊 Dataset Statistics")
                    
                    col_d, col_e, col_f = st.columns(3)
                    with col_d:
                        st.metric("Total Health Records", dataset_stats.get('total_records', 0))
                    with col_e:
                        st.metric("Unique Diseases", dataset_stats.get('unique_diseases', 0))
                    with col_f:
                        st.metric("Avg Response Length", f"{dataset_stats.get('avg_response_length', 0):.0f} chars")
                
                if model_history:
                    st.subheader("🤖 Model Training History")
                    history_df = pd.DataFrame(model_history)
                    st.dataframe(history_df[['model_name', 'accuracy', 'precision_score', 'recall_score', 'f1_score', 'training_date']])
                
                # Show current session chat history
                if st.session_state.chat_history:
                    st.subheader("💬 Current Session")
                    session_total = len(st.session_state.chat_history)
                    session_avg_conf = np.mean([conf for _, _, conf in st.session_state.chat_history if conf is not None])
                    
                    col_g, col_h = st.columns(2)
                    with col_g:
                        st.metric("Session Messages", session_total)
                    with col_h:
                        st.metric("Session Avg Confidence", f"{session_avg_conf:.2%}" if session_avg_conf else "N/A")
        
        with col3:
            if st.button("💾 Export Chat"):
                if st.session_state.chat_history:
                    # Create download data
                    chat_data = []
                    for user_msg, bot_msg, confidence in st.session_state.chat_history:
                        chat_data.append({
                            'User Question': user_msg,
                            'Bot Response': bot_msg,
                            'Confidence': confidence if confidence else 0.0
                        })
                    
                    df_chat = pd.DataFrame(chat_data)
                    csv = df_chat.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Chat History",
                        data=csv,
                        file_name=f"health_chat_history_{int(time.time())}.csv",
                        mime="text/csv"
                    )
            
    else:
        # Instructions when no model is trained
        st.info("👆 Please upload a CSV dataset and train a model, or use the sample dataset to start chatting!")
        
        # Expected CSV format
        st.subheader("📋 Expected CSV Format")
        st.write("Your CSV file should contain at least two columns:")
        
        example_df = create_sample_dataset()
        st.dataframe(example_df)
        
        # Feature highlights
        st.subheader("🌟 Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🤖 AI-Powered Responses**")
            st.write("- Machine learning-based health information")
            st.write("- Confidence scoring for each response")
            st.write("- Context-aware conversations")
            
        with col2:
            st.write("**📊 Data Analysis**")
            st.write("- Automatic data quality checking")
            st.write("- Training performance metrics")
            st.write("- Improvement suggestions")
        
        st.warning("⚠️ Remember: This is for informational purposes only and should not replace professional medical advice.")

if __name__ == "__main__":
    main()