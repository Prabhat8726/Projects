import re
import random
from datetime import datetime

class ChatHandler:
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer
        self.conversation_context = []
        self.fallback_responses = [
            "I'm not quite sure about that. Could you rephrase your question about health symptoms or conditions?",
            "I don't have enough information to answer that question accurately. Please ask about specific health symptoms or conditions.",
            "That's outside my area of expertise. I'm designed to help with general health information and symptoms.",
            "I'm sorry, I couldn't understand your question. Could you ask about a specific health concern or symptom?",
            "I'm not able to provide information on that topic. Please ask about health symptoms, conditions, or general wellness."
        ]
        
        # Common health-related keywords for better context understanding
        self.health_keywords = [
            'symptom', 'symptoms', 'pain', 'ache', 'fever', 'headache', 'cold', 'flu',
            'treatment', 'medicine', 'doctor', 'hospital', 'health', 'disease',
            'condition', 'illness', 'sick', 'hurt', 'sore', 'infection', 'cure',
            'therapy', 'medication', 'diagnosis', 'prevention', 'wellness', 'dizzy',
            'nausea', 'vomit', 'diarrhea', 'constipation', 'cough', 'sneeze',
            'allergy', 'rash', 'swelling', 'bruise', 'cut', 'wound', 'bleeding',
            'fatigue', 'tired', 'sleep', 'insomnia', 'anxiety', 'stress', 'depression',
            'chest', 'stomach', 'back', 'joint', 'muscle', 'bone', 'skin', 'eye',
            'ear', 'nose', 'throat', 'mouth', 'teeth', 'dental', 'mental', 'physical'
        ]
        
        # Emergency keywords that should trigger immediate medical attention advice
        self.emergency_keywords = [
            'emergency', 'urgent', 'severe', 'intense', 'unbearable', 'chest pain',
            'heart attack', 'stroke', 'bleeding', 'poisoning', 'overdose', 'suicide',
            'breathing', 'choking', 'unconscious', 'seizure', 'paralysis'
        ]
    
    def is_health_related(self, text):
        """Check if the question is health-related"""
        text_lower = text.lower()
        
        # Check for health keywords
        for keyword in self.health_keywords:
            if keyword in text_lower:
                return True
        
        # Check for common health question patterns
        health_patterns = [
            r'what.*(?:cause|symptom|treatment|cure)',
            r'how.*(?:treat|cure|prevent|manage)',
            r'why.*(?:hurt|pain|ache|sick)',
            r'is.*(?:normal|healthy|dangerous)',
            r'should.*(?:see|visit|consult).*doctor'
        ]
        
        for pattern in health_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def preprocess_question(self, question):
        """Preprocess user question for better understanding"""
        # Remove extra whitespace
        question = re.sub(r'\s+', ' ', question.strip())
        
        # Handle common question variations
        question = re.sub(r'\bwhat\'s\b', 'what is', question.lower())
        question = re.sub(r'\bhow\'s\b', 'how is', question.lower())
        question = re.sub(r'\bwhen\'s\b', 'when is', question.lower())
        
        return question
    
    def check_emergency_context(self, text):
        """Check if the question contains emergency-related keywords"""
        text_lower = text.lower()
        for keyword in self.emergency_keywords:
            if keyword in text_lower:
                return True
        return False
    
    def get_response(self, user_input):
        """Get response from the model with enhanced handling"""
        try:
            # Preprocess the input
            processed_input = self.preprocess_question(user_input)
            
            # Check for emergency situations first
            if self.check_emergency_context(processed_input):
                return self.handle_emergency_question(user_input)
            
            # Check if question is health-related
            if not self.is_health_related(processed_input):
                return self.handle_non_health_question(processed_input)
            
            # Get prediction from model
            response, confidence = self.model_trainer.predict(processed_input)
            
            # Handle low confidence responses
            if confidence < 0.3:
                return self.handle_low_confidence(user_input, response, confidence)
            
            # Enhance response with additional context
            enhanced_response = self.enhance_response(response, confidence)
            
            # Add to conversation context
            self.conversation_context.append({
                'user_input': user_input,
                'response': enhanced_response,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            return enhanced_response, confidence
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error processing your question. Please try rephrasing your health-related question."
            return error_response, 0.0
    
    def handle_emergency_question(self, question):
        """Handle emergency-related questions"""
        response = "🚨 **EMERGENCY ALERT** 🚨\n\n" \
                  "If you are experiencing a medical emergency, please:\n" \
                  "• Call emergency services immediately (911 in the US)\n" \
                  "• Go to the nearest emergency room\n" \
                  "• Contact your local emergency number\n\n" \
                  "This chatbot cannot provide emergency medical care. " \
                  "For life-threatening situations, seek immediate professional medical help."
        return response, 1.0
    
    def handle_non_health_question(self, question):
        """Handle non-health related questions"""
        response = "I'm a health assistant designed to help with medical symptoms and health-related questions. " \
                  "Please ask me about health symptoms, conditions, or general wellness topics."
        return response, 0.0
    
    def handle_low_confidence(self, original_question, response, confidence):
        """Handle low confidence predictions"""
        # Try to get similar responses
        try:
            similar_responses = self.model_trainer.get_similar_responses(original_question, top_k=3)
            
            if similar_responses and similar_responses[0][1] > 0.2:
                best_response = similar_responses[0][0]
                best_confidence = similar_responses[0][1]
                
                enhanced_response = f"I'm not entirely certain, but based on similar questions, here's what I can tell you: {best_response}\n\n" \
                                  f"Please consult with a healthcare professional for more specific advice."
                
                return enhanced_response, best_confidence
            else:
                fallback = random.choice(self.fallback_responses)
                return fallback, 0.0
                
        except Exception:
            fallback = random.choice(self.fallback_responses)
            return fallback, 0.0
    
    def enhance_response(self, response, confidence):
        """Enhance response with additional context and disclaimers"""
        enhanced = response
        
        # Add confidence-based disclaimers
        if confidence < 0.5:
            enhanced += "\n\n⚠️ This information is provided with lower confidence. Please consult a healthcare professional for accurate advice."
        elif confidence < 0.7:
            enhanced += "\n\n💡 This is general information. For personalized advice, please consult with a healthcare provider."
        
        # Add general medical disclaimer for all responses
        enhanced += "\n\n📋 Remember: This information is for educational purposes only and should not replace professional medical advice."
        
        return enhanced
    
    def get_conversation_summary(self):
        """Get a summary of the conversation"""
        if not self.conversation_context:
            return "No conversation history available."
        
        summary = f"Conversation Summary ({len(self.conversation_context)} interactions):\n\n"
        
        for i, interaction in enumerate(self.conversation_context[-5:], 1):  # Show last 5 interactions
            summary += f"{i}. Q: {interaction['user_input'][:50]}...\n"
            summary += f"   A: {interaction['response'][:100]}...\n"
            summary += f"   Confidence: {interaction['confidence']:.2%}\n\n"
        
        return summary
    
    def clear_context(self):
        """Clear conversation context"""
        self.conversation_context = []
    
    def get_context_length(self):
        """Get the length of conversation context"""
        return len(self.conversation_context)
    
    def suggest_questions(self):
        """Suggest health-related questions based on model capabilities"""
        if not self.model_trainer.is_trained:
            return []
        
        # Get some example responses to suggest related questions
        try:
            sample_responses = self.model_trainer.classes[:5]  # Get first 5 classes
            
            suggestions = []
            for response in sample_responses:
                # Create question suggestions based on response content
                if 'fever' in response.lower():
                    suggestions.append("What are the symptoms of fever?")
                elif 'headache' in response.lower():
                    suggestions.append("How can I treat a headache?")
                elif 'pain' in response.lower():
                    suggestions.append("What causes stomach pain?")
                elif 'stress' in response.lower():
                    suggestions.append("How can I manage stress?")
                else:
                    suggestions.append(f"Tell me about {response.split('.')[0].lower()}")
            
            return suggestions[:3]  # Return top 3 suggestions
            
        except Exception:
            return [
                "What are common symptoms of flu?",
                "How can I improve my sleep quality?",
                "What should I do for a headache?"
            ]
