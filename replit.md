# AI Health Chatbot

## Overview

This is a Streamlit-based AI-powered virtual health chatbot that uses machine learning to provide health-related information. The system allows users to train a custom model using their own health dataset and interact with an intelligent chatbot for health-related queries.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (July 18, 2025)

The AI Health Chatbot has been significantly enhanced with comprehensive improvements including database integration and enhanced datasets:

### Database Integration
- Added PostgreSQL database for persistent data storage
- Implemented chat history tracking and analytics
- Added model training history with metadata storage
- Created comprehensive analytics dashboard
- Integrated health data management system

### Enhanced Datasets
- Created comprehensive health dataset with 70+ detailed health topics
- Added specialized datasets for pediatric and geriatric health
- Implemented symptom-disease mapping functionality
- Enhanced responses with detailed, clinical-grade information
- Added treatment-focused datasets for specific conditions

### Advanced Model Training
- Enhanced TF-IDF vectorizer with better parameters (min_df, max_df, sublinear_tf)
- Improved logistic regression with class balancing and regularization
- Added comprehensive evaluation metrics (precision, recall, F1-score)
- Implemented confusion matrix visualization
- Added feature importance analysis
- Increased feature capacity for larger datasets (8000 features)

### Safety and Emergency Handling
- Implemented emergency keyword detection system
- Added immediate medical attention alerts for emergency situations
- Enhanced health-related keyword dictionary with 40+ terms
- Improved fallback responses for non-health queries

### Enhanced User Experience
- Added multiple dataset options (Sample, Enhanced, CSV Upload)
- Implemented suggested questions feature for better user guidance
- Added daily health tips in sidebar for educational value
- Improved color-coded confidence scoring in chat interface
- Created comprehensive analytics dashboard
- Added chat statistics and export functionality

### Data Processing Enhancements
- Added comprehensive CSV validation and analysis
- Implemented data quality checking with issue detection
- Added improvement suggestions based on dataset analysis
- Enhanced column analysis with sample values preview
- Integrated database storage for all health data

### Better Model Management
- Added timestamped model saving for version control
- Implemented model selection interface for loading existing models
- Enhanced model persistence with additional metadata
- Added training history tracking in database
- Created model performance comparison features

### Visualization Improvements
- Added confusion matrix heatmaps for model evaluation
- Enhanced metrics display with precision, recall, and F1-score
- Implemented feature importance visualization
- Added response class analysis display
- Created comprehensive analytics dashboard with charts

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Layout**: Wide layout with expandable sidebar
- **State Management**: Streamlit session state for maintaining chat history, model state, and component instances
- **UI Components**: File uploader, chat interface, metrics display, and training controls

### Backend Architecture
- **Model Training**: Scikit-learn based machine learning pipeline
- **Text Processing**: NLTK for natural language processing
- **Data Processing**: Pandas for data manipulation and cleaning
- **Chat Logic**: Custom chat handler with context management and fallback responses

### Machine Learning Pipeline
- **Vectorization**: TF-IDF vectorizer with n-gram support (1-2 grams)
- **Classification**: Logistic Regression classifier
- **Preprocessing**: Text cleaning, tokenization, lemmatization, and stopword removal
- **Evaluation**: Accuracy metrics, classification reports, and confusion matrices

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Primary Streamlit application entry point
- **Features**: Session state management, file upload handling, UI coordination
- **Integration**: Orchestrates all other components

### 2. Model Trainer (`model_trainer.py`)
- **Purpose**: Handles ML model training and prediction
- **Features**: 
  - TF-IDF vectorization with configurable parameters
  - Logistic regression classification
  - Model persistence using pickle
  - Comprehensive evaluation metrics
- **Text Processing**: NLTK-based preprocessing pipeline

### 3. Chat Handler (`chat_handler.py`)
- **Purpose**: Manages chat interactions and context
- **Features**:
  - Health-related question detection using keywords and regex patterns
  - Conversation context maintenance
  - Fallback responses for non-health queries
  - Question preprocessing and validation

### 4. Data Processor (`data_processor.py`)
- **Purpose**: Handles data cleaning and validation
- **Features**:
  - CSV data processing and validation
  - Data quality checks and statistics
  - Duplicate removal and missing value handling
  - Data format standardization

### 5. Utilities (`utils.py`)
- **Purpose**: Shared utility functions and UI components
- **Features**:
  - Medical disclaimer display
  - Training metrics visualization
  - Data statistics presentation
  - Reusable UI components

## Data Flow

1. **Data Upload**: User uploads CSV file with health questions and responses
2. **Data Processing**: System validates and cleans the uploaded data
3. **Model Training**: TF-IDF vectorizer processes text, LogisticRegression trains on the data
4. **Model Persistence**: Trained model is saved for future use
5. **Chat Interaction**: User queries are preprocessed, classified, and responses are generated
6. **Context Management**: Chat history is maintained in session state

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning pipeline
- **NLTK**: Natural language processing
- **Pickle**: Model serialization

### NLTK Resources
- **punkt**: Tokenization
- **stopwords**: Stop word removal
- **wordnet**: Lemmatization

### Visualization
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization

## Deployment Strategy

### Local Development
- **Environment**: Python with required dependencies
- **Execution**: Run via `streamlit run app.py`
- **Storage**: Local file system for model persistence

### Production Considerations
- **Model Storage**: Persistent storage for trained models
- **Session Management**: Streamlit's built-in session state
- **Data Security**: File upload validation and sanitization
- **Performance**: Model caching and efficient text processing

### Key Architectural Decisions

1. **Streamlit Framework**: Chosen for rapid prototyping and built-in web interface capabilities
2. **Scikit-learn Pipeline**: Selected for its robust ML tools and easy integration
3. **TF-IDF Vectorization**: Preferred over word embeddings for interpretability and performance
4. **Session State Management**: Utilized Streamlit's session state for maintaining application state
5. **Modular Design**: Separated concerns into distinct classes for maintainability
6. **Health-Focused Context**: Implemented keyword-based filtering to ensure health-related responses

The system is designed to be extensible, allowing for easy integration of additional ML models, data sources, or UI components while maintaining a clean separation of concerns.