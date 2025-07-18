import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

def show_medical_disclaimer():
    """Display medical disclaimer"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ Medical Disclaimer")
    st.sidebar.warning(
        "This chatbot provides general health information for educational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult with qualified healthcare professionals for medical concerns."
    )
    st.sidebar.markdown("---")

def display_metrics(training_results: Dict[str, Any]):
    """Display training metrics in a formatted way"""
    st.subheader("📊 Training Results")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{training_results['accuracy']:.2%}")
    
    with col2:
        st.metric("Precision", f"{training_results.get('precision', 0):.2%}")
    
    with col3:
        st.metric("Recall", f"{training_results.get('recall', 0):.2%}")
    
    with col4:
        st.metric("F1-Score", f"{training_results.get('f1_score', 0):.2%}")
    
    # Additional metrics
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("Training Samples", training_results['train_size'])
    
    with col6:
        st.metric("Test Samples", training_results['test_size'])
    
    with col7:
        st.metric("Features", training_results['num_features'])
    
    # Display confusion matrix if available
    if 'confusion_matrix' in training_results and 'class_names' in training_results:
        with st.expander("🔍 Confusion Matrix"):
            try:
                cm = np.array(training_results['confusion_matrix'])
                class_names = training_results['class_names']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names, yticklabels=class_names, ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error displaying confusion matrix: {str(e)}")
    
    # Display feature importance if available
    if 'feature_names' in training_results:
        with st.expander("🔤 Important Features"):
            st.write("Top features used by the model:")
            features = training_results['feature_names']
            for i, feature in enumerate(features[:20], 1):
                st.write(f"{i}. {feature}")
    
    # Display response classes
    if 'class_names' in training_results:
        with st.expander("📋 Response Classes"):
            st.write(f"Model can predict {len(training_results['class_names'])} different types of responses:")
            for i, class_name in enumerate(training_results['class_names'], 1):
                st.write(f"{i}. {class_name[:100]}{'...' if len(class_name) > 100 else ''}")
    
    # Detailed classification report
    if st.expander("📋 Detailed Classification Report"):
        try:
            report = training_results['classification_report']
            
            # Convert to DataFrame for better display
            df_report = pd.DataFrame(report).transpose()
            
            # Format numerical columns
            for col in ['precision', 'recall', 'f1-score']:
                if col in df_report.columns:
                    df_report[col] = df_report[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
            
            # Display support as integers
            if 'support' in df_report.columns:
                df_report['support'] = df_report['support'].apply(lambda x: f"{int(x)}" if isinstance(x, (int, float)) else x)
            
            st.dataframe(df_report)
            
        except Exception as e:
            st.error(f"Error displaying classification report: {str(e)}")

def display_data_stats(data_stats: Dict[str, Any]):
    """Display data statistics"""
    st.subheader("📈 Dataset Statistics")
    
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", data_stats['total_samples'])
    
    with col2:
        st.metric("Unique Responses", data_stats['unique_responses'])
    
    with col3:
        st.metric("Avg Question Length", f"{data_stats['avg_question_length']:.1f} words")
    
    with col4:
        st.metric("Avg Response Length", f"{data_stats['avg_response_length']:.1f} words")
    
    # Length distribution
    if st.expander("📊 Length Distribution"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Question Lengths")
            st.write(f"Min: {data_stats['min_question_length']} words")
            st.write(f"Max: {data_stats['max_question_length']} words")
        
        with col2:
            st.subheader("Response Lengths")
            st.write(f"Min: {data_stats['min_response_length']} words")
            st.write(f"Max: {data_stats['max_response_length']} words")
    
    # Response distribution
    if st.expander("📊 Response Distribution"):
        response_dist = data_stats['response_distribution']
        if response_dist:
            st.subheader("Most Common Responses")
            
            # Create a bar chart
            responses = list(response_dist.keys())[:10]
            counts = list(response_dist.values())[:10]
            
            # Truncate long responses for display
            truncated_responses = [resp[:50] + "..." if len(resp) > 50 else resp for resp in responses]
            
            chart_data = pd.DataFrame({
                'Response': truncated_responses,
                'Count': counts
            })
            
            st.bar_chart(chart_data.set_index('Response'))
        else:
            st.info("No response distribution data available")

def display_data_issues(issues: List[str]):
    """Display data quality issues"""
    if issues:
        st.subheader("⚠️ Data Quality Issues")
        for issue in issues:
            st.warning(f"• {issue}")
    else:
        st.success("✅ No major data quality issues detected")

def display_improvement_suggestions(suggestions: List[str]):
    """Display improvement suggestions"""
    if suggestions:
        st.subheader("💡 Improvement Suggestions")
        for suggestion in suggestions:
            st.info(f"• {suggestion}")

def format_confidence_color(confidence: float) -> str:
    """Return color based on confidence level"""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"

def create_confidence_badge(confidence: float) -> str:
    """Create a colored confidence badge"""
    color = format_confidence_color(confidence)
    return f"<span style='color: {color}; font-weight: bold;'>{confidence:.1%}</span>"

def export_chat_history(chat_history: List[tuple], filename: str = "chat_history.csv"):
    """Export chat history to CSV"""
    try:
        data = []
        for user_msg, bot_msg, confidence in chat_history:
            data.append({
                'User Question': user_msg,
                'Bot Response': bot_msg,
                'Confidence': confidence if confidence else 0.0
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return True
        
    except Exception:
        return False

def validate_csv_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate CSV structure and return analysis"""
    analysis = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'suggestions': [],
        'column_info': {}
    }
    
    # Check if dataframe is empty
    if df.empty:
        analysis['is_valid'] = False
        analysis['errors'].append("CSV file is empty")
        return analysis
    
    # Check minimum columns
    if len(df.columns) < 2:
        analysis['is_valid'] = False
        analysis['errors'].append("CSV must have at least 2 columns")
        return analysis
    
    # Analyze each column
    for col in df.columns:
        col_info = {
            'name': col,
            'type': str(df[col].dtype),
            'non_null_count': df[col].notna().sum(),
            'null_count': df[col].isna().sum(),
            'unique_count': df[col].nunique(),
            'sample_values': df[col].dropna().head(3).tolist()
        }
        
        analysis['column_info'][col] = col_info
        
        # Check for high null percentage
        null_percentage = (col_info['null_count'] / len(df)) * 100
        if null_percentage > 20:
            analysis['warnings'].append(f"Column '{col}' has {null_percentage:.1f}% missing values")
    
    # Check for potential question/response columns
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_length = df[col].dropna().str.len().mean()
            if avg_length > 10:  # Likely text column
                text_columns.append((col, avg_length))
    
    if len(text_columns) < 2:
        analysis['warnings'].append("Could not identify suitable question and response columns")
    
    return analysis

def create_sample_dataset() -> pd.DataFrame:
    """Create a sample health dataset for demonstration"""
    sample_data = {
        'question': [
            "What are the symptoms of common cold?",
            "How to treat a headache?",
            "What causes stomach pain?",
            "How to manage stress and anxiety?",
            "What are the signs of dehydration?",
            "How to improve sleep quality?",
            "What to do for muscle cramps?",
            "How to prevent allergies?",
            "What are the symptoms of flu?",
            "How to boost immune system?",
            "What should I do for back pain?",
            "How to deal with insomnia?",
            "What are the causes of dizziness?",
            "How to treat a sore throat?",
            "What to do for food poisoning?",
            "How to manage high blood pressure?",
            "What are the symptoms of diabetes?",
            "How to treat minor cuts and wounds?",
            "What causes frequent urination?",
            "How to deal with acid reflux?"
        ],
        'response': [
            "Common cold symptoms include runny nose, sneezing, cough, sore throat, and mild fatigue. Rest and hydration are key.",
            "For headaches, try rest, hydration, over-the-counter pain relievers, and stress management techniques.",
            "Stomach pain can result from indigestion, gas, stress, or food poisoning. Try bland foods and stay hydrated.",
            "Manage stress through regular exercise, meditation, deep breathing, adequate sleep, and seeking support when needed.",
            "Dehydration signs include dry mouth, fatigue, dizziness, dark urine, and decreased urination. Drink fluids gradually.",
            "Improve sleep by maintaining a regular schedule, creating a comfortable environment, and avoiding screens before bed.",
            "For muscle cramps, gently stretch the muscle, apply heat or cold, massage the area, and ensure proper hydration.",
            "Prevent allergies by avoiding triggers, keeping indoor air clean, and considering antihistamines when necessary.",
            "Flu symptoms include fever, body aches, fatigue, cough, and chills. Rest, fluids, and antiviral medications may help.",
            "Boost immunity with balanced nutrition, regular exercise, adequate sleep, stress management, and good hygiene practices.",
            "For back pain, try rest, gentle stretching, heat therapy, and over-the-counter pain relievers. Avoid heavy lifting.",
            "Combat insomnia with a consistent sleep schedule, comfortable bedroom environment, and relaxation techniques before bed.",
            "Dizziness can be caused by dehydration, low blood sugar, inner ear problems, or medication side effects. Stay hydrated.",
            "Treat sore throat with warm salt water gargles, throat lozenges, honey, and plenty of fluids. Rest your voice.",
            "Food poisoning symptoms include nausea, vomiting, and diarrhea. Stay hydrated and eat bland foods when tolerated.",
            "Manage high blood pressure through regular exercise, low-sodium diet, stress management, and prescribed medications.",
            "Diabetes symptoms include excessive thirst, frequent urination, fatigue, and unexplained weight loss. See a doctor.",
            "Clean minor cuts with water, apply antibiotic ointment, and cover with a bandage. Watch for signs of infection.",
            "Frequent urination can indicate diabetes, UTI, or overactive bladder. Consult a healthcare provider for evaluation.",
            "Manage acid reflux by avoiding trigger foods, eating smaller meals, and elevating your head while sleeping."
        ]
    }
    
    return pd.DataFrame(sample_data)

def get_health_tips() -> List[str]:
    """Get random health tips"""
    tips = [
        "💧 Stay hydrated by drinking 8-10 glasses of water daily",
        "🚶‍♀️ Aim for at least 30 minutes of physical activity daily",
        "😴 Get 7-9 hours of quality sleep each night",
        "🥗 Eat a balanced diet rich in fruits and vegetables",
        "🧘‍♀️ Practice stress management techniques like meditation",
        "👥 Maintain social connections for mental health",
        "🚭 Avoid smoking and limit alcohol consumption",
        "🏥 Schedule regular health check-ups",
        "🧼 Practice good hygiene, especially hand washing",
        "☀️ Get some sunlight for vitamin D"
    ]
    
    import random
    return random.sample(tips, 3)
