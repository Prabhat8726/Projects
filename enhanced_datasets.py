import pandas as pd
import numpy as np
import requests
import io
from typing import Dict, List, Optional
import random
import json

class EnhancedHealthDatasets:
    def __init__(self):
        self.datasets = {}
        self.comprehensive_health_data = self._create_comprehensive_dataset()
    
    def _create_comprehensive_dataset(self) -> pd.DataFrame:
        """Create a comprehensive health dataset with detailed responses"""
        
        # Create a balanced dataset with matching questions and responses
        questions = [
            "What are the symptoms of asthma?",
            "How do I manage chronic bronchitis?",
            "What causes persistent cough?",
            "How to treat pneumonia symptoms?",
            "What are the symptoms of high blood pressure?",
            "How to manage heart disease?",
            "What causes chest pain?",
            "How to prevent heart attacks?",
            "How to treat acid reflux?",
            "What causes stomach ulcers?",
            "How to manage irritable bowel syndrome?",
            "What are the symptoms of food poisoning?",
            "How to manage anxiety and panic attacks?",
            "What are the symptoms of depression?",
            "How to cope with chronic stress?",
            "What causes sleep disorders?",
            "How to treat chronic back pain?",
            "What causes joint pain and stiffness?",
            "How to manage arthritis symptoms?",
            "What are the signs of muscle strain?",
            "How to manage diabetes effectively?",
            "What are the symptoms of thyroid disorders?",
            "How to balance hormones naturally?",
            "What causes excessive fatigue?",
            "How to boost immune system naturally?",
            "What causes frequent infections?",
            "How to manage autoimmune conditions?",
            "What are the symptoms of allergies?",
            "How to improve sleep quality?",
            "What causes insomnia?",
            "How to manage sleep apnea?",
            "How to maintain good eye health?",
            "What causes dry eyes?",
            "How to prevent vision problems?",
            "How to maintain oral hygiene?",
            "What causes tooth sensitivity?",
            "How to prevent gum disease?",
            "How often should I get health checkups?",
            "What vaccinations do adults need?",
            "How to maintain healthy lifestyle?"
        ]
        
        responses = [
            "Asthma symptoms include wheezing, shortness of breath, chest tightness, and persistent cough, especially at night or early morning. These symptoms occur when airways become inflamed and narrowed. Triggers can include allergens like dust mites, pollen, pet dander, smoke, cold air, exercise, or stress. Management involves identifying and avoiding triggers, using prescribed inhalers (bronchodilators for quick relief and corticosteroids for long-term control), maintaining a healthy weight, and having an asthma action plan. During an attack, use your rescue inhaler immediately, sit upright, breathe slowly and deeply, and seek emergency care if symptoms don't improve.",
            
            "Chronic bronchitis management focuses on reducing inflammation, clearing mucus, and preventing complications. This includes quitting smoking (most important), avoiding air pollutants and irritants, staying hydrated to thin mucus, using prescribed bronchodilators and anti-inflammatory medications, practicing breathing exercises, and getting regular exercise as tolerated. Pulmonary rehabilitation programs can teach breathing techniques and provide structured exercise training. Annual flu shots and pneumonia vaccines are crucial. Monitor for signs of infection and seek prompt medical attention. Oxygen therapy may be needed in advanced cases.",
            
            "Persistent cough can result from multiple causes including post-nasal drip, asthma, GERD (acid reflux), chronic bronchitis, medications (ACE inhibitors), or infections. Treatment depends on the underlying cause: antihistamines for allergies, acid reducers for GERD, bronchodilators for asthma, or antibiotics for bacterial infections. Home remedies include honey (for adults), staying hydrated, using a humidifier, avoiding irritants, and throat lozenges. See a doctor if cough persists over 3 weeks, produces blood, or is accompanied by fever, weight loss, or difficulty breathing.",
            
            "Pneumonia symptoms include fever, chills, cough with phlegm, chest pain, shortness of breath, and fatigue. Treatment varies by type: bacterial pneumonia requires antibiotics, viral pneumonia needs supportive care, and fungal pneumonia requires antifungal medications. Rest is crucial - get plenty of sleep and avoid strenuous activities. Stay hydrated with water, herbal teas, and clear broths. Use a humidifier or breathe steam from a hot shower. Take fever reducers as directed. Seek immediate medical care if you experience severe difficulty breathing, chest pain, high fever, or confusion.",
            
            "High blood pressure symptoms are often silent, which is why it's called the 'silent killer.' Some people may experience headaches, shortness of breath, dizziness, or nosebleeds, but these usually occur only when blood pressure reaches dangerously high levels. Management includes lifestyle changes: maintaining a healthy weight, exercising regularly (150 minutes of moderate activity weekly), following a low-sodium diet rich in fruits and vegetables (DASH diet), limiting alcohol, quitting smoking, managing stress through relaxation techniques, and getting adequate sleep. Medications may include ACE inhibitors, diuretics, beta-blockers, or calcium channel blockers.",
            
            "Heart disease management involves comprehensive lifestyle changes and medical care. This includes following a heart-healthy diet low in saturated fats and high in omega-3 fatty acids, exercising regularly as approved by your doctor, maintaining a healthy weight, managing stress, quitting smoking, limiting alcohol, and taking prescribed medications consistently. Regular check-ups monitor cholesterol levels, blood pressure, and heart function. Medications may include statins for cholesterol, blood thinners, ACE inhibitors, or beta-blockers. Cardiac rehabilitation programs provide structured exercise training and education.",
            
            "Chest pain can have various causes including heart problems (angina, heart attack), lung issues (pneumonia, pleurisy), digestive problems (acid reflux, gallbladder), or musculoskeletal strain. Heart-related chest pain typically feels like pressure, squeezing, or tightness that may radiate to the arm, jaw, or back. Seek immediate emergency care if you experience severe chest pain, especially with shortness of breath, nausea, sweating, or dizziness. For non-emergency chest pain, treatment depends on the cause: antacids for reflux, anti-inflammatories for muscle strain, or cardiac medications for heart conditions.",
            
            "Heart attack prevention involves controlling risk factors: maintaining healthy cholesterol levels (LDL below 100 mg/dL), keeping blood pressure under control, managing diabetes, maintaining a healthy weight, exercising regularly, not smoking, eating a heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins, limiting saturated fats and trans fats, managing stress, and getting adequate sleep. Know the warning signs: chest pain or discomfort, shortness of breath, pain in arms, back, neck, or jaw, nausea, and cold sweats. Take prescribed medications consistently and have regular check-ups with your healthcare provider.",
            
            "Acid reflux management includes dietary changes: avoiding trigger foods (spicy, acidic, fatty foods, chocolate, caffeine, alcohol), eating smaller meals, not lying down within 3 hours of eating, maintaining a healthy weight, elevating the head of your bed 6-8 inches, and wearing loose-fitting clothes. Medications include antacids for quick relief, H2 blockers (famotidine) for longer-lasting relief, and proton pump inhibitors (omeprazole) for severe cases. Lifestyle changes: quit smoking, manage stress, and chew gum after meals to increase saliva production. Avoid late-night eating and stay upright after meals.",
            
            "Stomach ulcers are usually caused by H. pylori bacteria infection or NSAIDs (non-steroidal anti-inflammatory drugs). Symptoms include burning stomach pain, bloating, heartburn, and nausea. Treatment involves antibiotics for H. pylori (triple therapy), proton pump inhibitors to reduce acid production, and avoiding NSAIDs. Dietary recommendations include eating smaller, frequent meals, avoiding spicy and acidic foods, limiting alcohol and caffeine, and including probiotic foods. Stress management and smoking cessation are important. Complications like bleeding or perforation require immediate medical attention.",
            
            "IBS (Irritable Bowel Syndrome) management focuses on dietary modifications, stress management, and symptom control. Follow a low-FODMAP diet initially, then gradually reintroduce foods to identify triggers. Common triggers include high-fat foods, caffeine, alcohol, artificial sweeteners, and certain fruits and vegetables. Eat regular meals, stay hydrated, and include fiber gradually. Stress management through relaxation techniques, exercise, and adequate sleep is crucial. Medications may include antispasmodics, fiber supplements, or specific IBS medications. Probiotics may help some people.",
            
            "Food poisoning symptoms include nausea, vomiting, diarrhea, abdominal cramps, fever, and dehydration. Most cases resolve within 3-7 days with supportive care. Stay hydrated with clear fluids, electrolyte solutions, or ice chips. Rest and avoid solid foods initially, then gradually introduce bland foods (BRAT diet: bananas, rice, applesauce, toast). Avoid dairy, caffeine, alcohol, and fatty foods until recovered. Seek medical attention if you experience severe dehydration, high fever, bloody stools, or symptoms lasting more than several days.",
            
            "Anxiety and panic attack management involves learning coping strategies, lifestyle changes, and sometimes medication. During a panic attack, practice deep breathing (breathe in for 4 counts, hold for 4, exhale for 4), use grounding techniques (5-4-3-2-1 method: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste), and remind yourself that the attack will pass. Long-term management includes regular exercise, limiting caffeine and alcohol, practicing mindfulness or meditation, maintaining a regular sleep schedule, and learning stress management techniques.",
            
            "Depression symptoms include persistent sadness, loss of interest in activities, changes in appetite or sleep, fatigue, difficulty concentrating, feelings of worthlessness, and thoughts of self-harm. Treatment typically involves therapy (cognitive behavioral therapy, interpersonal therapy), medication (antidepressants), and lifestyle changes. Important lifestyle factors include regular exercise (as effective as medication for mild depression), maintaining social connections, eating a balanced diet, getting adequate sleep, practicing stress management, and avoiding alcohol and drugs. If you have thoughts of self-harm, seek immediate professional help.",
            
            "Chronic stress management requires a multifaceted approach: identify stress triggers, develop healthy coping strategies, practice relaxation techniques (deep breathing, progressive muscle relaxation, meditation), maintain regular exercise, prioritize sleep, eat a balanced diet, limit caffeine and alcohol, and maintain social connections. Time management and setting boundaries are crucial. Consider therapy to develop coping skills and address underlying issues. Regular breaks, hobbies, and activities you enjoy help build resilience.",
            
            "Sleep disorders include insomnia, sleep apnea, restless leg syndrome, and narcolepsy. Symptoms may include difficulty falling asleep, frequent awakenings, snoring, daytime fatigue, or excessive sleepiness. Good sleep hygiene includes maintaining a consistent sleep schedule, creating a comfortable sleep environment, avoiding screens before bedtime, limiting caffeine and alcohol, and establishing a relaxing bedtime routine. Treatment depends on the specific disorder and may include lifestyle changes, CPAP therapy for sleep apnea, medications, or cognitive behavioral therapy for insomnia.",
            
            "Chronic back pain management involves a comprehensive approach including physical therapy, exercise, posture correction, and pain management techniques. Strengthen core muscles through targeted exercises, practice good posture, use ergonomic furniture, and avoid prolonged sitting or standing. Apply heat or cold therapy as appropriate. Stress management and adequate sleep are crucial as they affect pain perception. Medications may include anti-inflammatories, muscle relaxants, or pain relievers. Alternative therapies like acupuncture, massage, or chiropractic care may help.",
            
            "Joint pain and stiffness can result from arthritis, injury, overuse, or inflammatory conditions. Common causes include osteoarthritis (wear and tear), rheumatoid arthritis (autoimmune), gout, or fibromyalgia. Treatment depends on the underlying cause but may include rest, ice or heat therapy, gentle exercise, weight management, anti-inflammatory medications, and physical therapy. Maintain joint mobility through regular, low-impact activities like swimming or walking. Avoid prolonged inactivity and protect joints from excessive stress.",
            
            "Arthritis symptoms management involves a combination of medical treatment, lifestyle modifications, and self-care strategies. This includes taking prescribed medications (NSAIDs, DMARDs, or biologics), maintaining a healthy weight to reduce joint stress, engaging in regular low-impact exercise, using heat and cold therapy, practicing stress management, and getting adequate sleep. Physical therapy can help maintain joint function and range of motion. Occupational therapy may provide adaptive strategies for daily activities. Support groups can offer emotional support and practical tips.",
            
            "Muscle strain signs include sudden pain, tenderness, swelling, bruising, and limited range of motion. Acute treatment follows the RICE protocol: Rest the injured muscle, Ice for 15-20 minutes every 2-3 hours for the first 48 hours, Compression with an elastic bandage, and Elevation if possible. After 48 hours, gentle stretching and gradual return to activity can help. Over-the-counter pain relievers may provide relief. Seek medical attention if pain is severe, if you can't move the muscle, or if numbness or weakness develops.",
            
            "Diabetes management requires consistent blood sugar monitoring, medication adherence, dietary planning, and regular exercise. Follow a balanced diet with controlled carbohydrates, focusing on whole grains, lean proteins, healthy fats, and plenty of vegetables. Monitor blood glucose levels as directed, take medications consistently, and maintain a healthy weight. Regular physical activity helps improve insulin sensitivity. Get regular check-ups for complications screening (eyes, feet, kidneys). Learn to recognize and treat hypoglycemia and hyperglycemia. Work with a healthcare team including a dietitian and diabetes educator.",
            
            "Thyroid disorder symptoms vary depending on whether the thyroid is overactive (hyperthyroidism) or underactive (hypothyroidism). Hyperthyroidism symptoms include rapid heartbeat, weight loss, nervousness, heat intolerance, and excessive sweating. Hypothyroidism symptoms include fatigue, weight gain, cold intolerance, dry skin, and depression. Both conditions require medical evaluation and treatment. Treatment may include medications to regulate thyroid hormone levels, radioactive iodine therapy, or surgery in some cases. Regular monitoring and medication adjustments are often necessary.",
            
            "Hormone balance can be supported through lifestyle modifications including maintaining a healthy weight, eating a balanced diet rich in whole foods, getting regular exercise, managing stress, ensuring adequate sleep, and limiting exposure to endocrine disruptors. Include foods rich in omega-3 fatty acids, fiber, and antioxidants. Consider reducing processed foods, sugar, and excess caffeine. Stress management techniques like meditation, yoga, or deep breathing can help regulate cortisol levels. Consult with a healthcare provider for hormonal imbalances that may require medical intervention.",
            
            "Excessive fatigue can result from various causes including sleep disorders, anemia, thyroid problems, depression, chronic fatigue syndrome, or underlying medical conditions. Evaluation should include assessment of sleep quality, stress levels, diet, exercise habits, and medical history. Treatment depends on the underlying cause but may include improving sleep hygiene, addressing nutritional deficiencies, managing stress, treating underlying conditions, and gradually increasing physical activity. Persistent fatigue that interferes with daily activities should be evaluated by a healthcare provider.",
            
            "Immune system support involves maintaining a healthy lifestyle with adequate sleep (7-9 hours), regular exercise, stress management, and proper nutrition. Eat a varied diet rich in fruits, vegetables, whole grains, and lean proteins. Key nutrients include vitamin C, vitamin D, zinc, and probiotics. Stay hydrated, limit alcohol, don't smoke, and practice good hygiene. Manage stress through relaxation techniques, social connections, and activities you enjoy. Get recommended vaccinations and avoid overuse of antibiotics. Regular handwashing and avoiding close contact with sick individuals help prevent infections.",
            
            "Frequent infections may indicate a weakened immune system, which can result from chronic stress, poor nutrition, lack of sleep, underlying medical conditions, or certain medications. Evaluation should include assessment of lifestyle factors, medical history, and possibly laboratory tests to check immune function. Treatment focuses on addressing underlying causes, improving nutrition, managing stress, ensuring adequate sleep, and maintaining good hygiene practices. Recurrent infections that are severe or unusual should be evaluated by a healthcare provider to rule out immune system disorders.",
            
            "Autoimmune condition management involves working with healthcare providers to develop a comprehensive treatment plan that may include immunosuppressive medications, anti-inflammatory drugs, and lifestyle modifications. Important lifestyle factors include maintaining a healthy diet, getting regular exercise as tolerated, managing stress, ensuring adequate sleep, and avoiding known triggers. Some people benefit from elimination diets to identify food sensitivities. Regular monitoring and medication adjustments are often necessary. Support groups can provide emotional support and practical advice for managing chronic conditions.",
            
            "Allergy symptoms include sneezing, runny nose, itchy eyes, skin rashes, and in severe cases, difficulty breathing. Common allergens include pollen, dust mites, pet dander, certain foods, and medications. Management involves identifying and avoiding triggers when possible, using antihistamines for symptom relief, and considering allergy shots (immunotherapy) for severe cases. For food allergies, strict avoidance is crucial and carrying emergency medications like epinephrine may be necessary. Environmental controls like air purifiers and allergen-proof bedding can help reduce exposure.",
            
            "Sleep quality improvement involves establishing good sleep hygiene practices: maintaining a consistent sleep schedule, creating a comfortable sleep environment (cool, dark, quiet), avoiding screens before bedtime, limiting caffeine and alcohol, and establishing a relaxing bedtime routine. Regular exercise during the day can improve sleep quality, but avoid vigorous exercise close to bedtime. If you have trouble falling asleep, try relaxation techniques like deep breathing or meditation. Persistent sleep problems that affect daily functioning should be evaluated by a healthcare provider.",
            
            "Insomnia can be caused by stress, anxiety, depression, medical conditions, medications, or poor sleep habits. Treatment may include cognitive behavioral therapy for insomnia (CBT-I), which is often more effective than medication. Sleep hygiene improvements, stress management, and addressing underlying causes are important. Short-term use of sleep medications may be helpful in some cases, but long-term use should be avoided due to potential dependence and side effects. If insomnia persists despite lifestyle changes, consult with a healthcare provider for evaluation and treatment options.",
            
            "Sleep apnea is a condition where breathing repeatedly stops and starts during sleep, leading to poor sleep quality and daytime fatigue. Symptoms include loud snoring, gasping during sleep, morning headaches, and excessive daytime sleepiness. Treatment options include lifestyle changes (weight loss, sleeping on side), CPAP (continuous positive airway pressure) therapy, oral appliances, or surgery in severe cases. Untreated sleep apnea can increase the risk of high blood pressure, heart problems, and stroke. A sleep study is typically needed for diagnosis.",
            
            "Good eye health maintenance involves protecting your eyes from UV radiation with sunglasses, eating a diet rich in omega-3 fatty acids and antioxidants, not smoking, taking regular breaks from screen time (20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds), maintaining proper lighting when reading or working, and getting regular eye exams. Stay hydrated and manage chronic conditions like diabetes and high blood pressure that can affect eye health. Wear protective eyewear during sports or activities that pose eye injury risks.",
            
            "Dry eyes can result from decreased tear production, increased tear evaporation, or poor tear quality. Causes include aging, hormonal changes, medications, environmental factors, or underlying conditions. Treatment includes using artificial tears, avoiding dry environments, taking breaks from screen time, staying hydrated, and using a humidifier. Warm compresses and gentle eyelid massage may help. Prescription medications or procedures may be necessary for severe cases. Avoid rubbing your eyes and protect them from wind and dust.",
            
            "Vision problem prevention involves regular eye exams, protecting eyes from UV radiation, maintaining a healthy diet rich in vitamins A, C, and E, omega-3 fatty acids, and zinc, not smoking, managing chronic conditions like diabetes and hypertension, and practicing good eye hygiene. Wear protective eyewear when necessary, maintain proper lighting for reading and computer work, and take regular breaks from close-up tasks. Early detection through regular eye exams is crucial for preventing vision loss from conditions like glaucoma, macular degeneration, and diabetic retinopathy.",
            
            "Oral hygiene maintenance involves brushing teeth twice daily with fluoride toothpaste, flossing daily, using an antimicrobial mouthwash, and getting regular dental cleanings and checkups. Limit sugary and acidic foods and drinks, don't smoke or use tobacco products, and stay hydrated. Replace your toothbrush every 3-4 months or after illness. Consider using a tongue scraper to remove bacteria and freshen breath. If you have specific dental concerns like gum disease or tooth sensitivity, work with your dentist to develop an appropriate treatment plan.",
            
            "Tooth sensitivity can result from worn enamel, exposed tooth roots, tooth decay, or dental procedures. Management includes using desensitizing toothpaste, avoiding acidic foods and drinks, using a soft-bristled toothbrush, and not brushing immediately after eating acidic foods. Fluoride treatments, dental sealants, or other dental procedures may be necessary for severe cases. Practice good oral hygiene and see your dentist regularly to prevent further enamel loss and address underlying causes of sensitivity.",
            
            "Gum disease prevention involves maintaining excellent oral hygiene through regular brushing and flossing, getting professional dental cleanings, not smoking, eating a balanced diet, and managing conditions like diabetes that can increase gum disease risk. Early signs of gum disease include bleeding gums, bad breath, and gum recession. If caught early (gingivitis), gum disease is reversible with proper care. Advanced gum disease (periodontitis) requires professional treatment and ongoing maintenance to prevent tooth loss and other complications.",
            
            "Health checkup frequency depends on age, risk factors, and existing health conditions. Generally, adults should have annual checkups that include blood pressure monitoring, cholesterol screening, and cancer screenings as appropriate for age and risk factors. Specific screenings like mammograms, colonoscopies, and bone density tests have recommended schedules based on age and risk factors. Your healthcare provider can help determine the appropriate frequency of checkups and screenings based on your individual health profile and family history.",
            
            "Adult vaccination recommendations include annual flu shots, tetanus-diphtheria boosters every 10 years, and other vaccines based on age, health conditions, and risk factors. These may include pneumococcal vaccines, shingles vaccine, hepatitis vaccines, and HPV vaccine for certain age groups. Travel vaccines may be necessary for international travel. Consult with your healthcare provider to ensure you're up to date on recommended vaccines based on your age, health status, and lifestyle factors.",
            
            "Healthy lifestyle maintenance involves eating a balanced diet rich in fruits, vegetables, whole grains, and lean proteins, engaging in regular physical activity (150 minutes of moderate exercise weekly), maintaining a healthy weight, not smoking, limiting alcohol consumption, getting adequate sleep, managing stress, and maintaining social connections. Regular health screenings and preventive care are important for early detection and management of health conditions. Stay hydrated, practice good hygiene, and avoid risky behaviors that can negatively impact health."
        ]
        
        # Ensure we have matching arrays
        health_data = {
            'question': questions,
            'response': responses
        }
        
        return pd.DataFrame(health_data)
    

    
    def get_comprehensive_dataset(self) -> pd.DataFrame:
        """Get the comprehensive health dataset"""
        return self.comprehensive_health_data
    
    def get_symptom_disease_dataset(self) -> pd.DataFrame:
        """Create a symptom-disease mapping dataset"""
        symptom_disease_data = {
            'symptoms': [
                'fever, cough, body aches, fatigue',
                'chest pain, shortness of breath, sweating',
                'headache, nausea, sensitivity to light',
                'joint pain, stiffness, swelling',
                'abdominal pain, bloating, changes in bowel habits',
                'persistent sadness, loss of interest, fatigue',
                'excessive thirst, frequent urination, blurred vision',
                'wheezing, chest tightness, difficulty breathing',
                'skin rash, itching, redness',
                'dizziness, balance problems, hearing changes'
            ],
            'disease_category': [
                'Viral Infection',
                'Cardiovascular',
                'Neurological',
                'Rheumatologic',
                'Gastrointestinal',
                'Mental Health',
                'Endocrine',
                'Respiratory',
                'Dermatological',
                'Otolaryngological'
            ],
            'detailed_response': [
                'Viral infections like flu present with fever, cough, body aches, and fatigue. Treatment focuses on rest, hydration, and symptom management. Most viral infections resolve within 7-10 days. Seek medical care if symptoms worsen or persist.',
                'Cardiovascular symptoms require immediate attention. Chest pain with shortness of breath and sweating may indicate heart attack. Call emergency services immediately. Risk factors include high blood pressure, diabetes, and smoking.',
                'Migraines cause severe headaches with nausea and light sensitivity. Triggers include stress, certain foods, hormonal changes, and sleep disruption. Treatment includes pain medications, rest in a dark room, and preventive strategies.',
                'Arthritis causes joint pain, stiffness, and swelling. Management includes anti-inflammatory medications, physical therapy, exercise, and heat/cold therapy. Maintaining a healthy weight reduces joint stress.',
                'Gastrointestinal issues like IBS cause abdominal pain, bloating, and bowel habit changes. Management includes dietary modifications, stress reduction, and appropriate medications based on symptoms.',
                'Depression presents with persistent sadness, loss of interest, and fatigue. Treatment includes therapy, medication, lifestyle changes, and social support. Professional help is important for proper management.',
                'Diabetes symptoms include excessive thirst, frequent urination, and blurred vision. Management requires blood sugar monitoring, medication, dietary control, and regular exercise. Complications can be serious if untreated.',
                'Asthma causes wheezing, chest tightness, and breathing difficulty. Treatment includes bronchodilators for quick relief and controller medications for long-term management. Trigger avoidance is crucial.',
                'Skin conditions like eczema cause rash, itching, and redness. Treatment includes moisturizers, topical medications, and trigger identification. Proper skin care routine is essential.',
                'Ear problems can cause dizziness, balance issues, and hearing changes. Treatment depends on the cause - infections require antibiotics, while balance disorders may need specialized therapy.'
            ]
        }
        
        return pd.DataFrame(symptom_disease_data)
    
    def get_treatment_dataset(self) -> pd.DataFrame:
        """Create a treatment-focused dataset"""
        treatment_data = {
            'condition': [
                'Hypertension',
                'Type 2 Diabetes',
                'Anxiety',
                'Osteoarthritis',
                'GERD',
                'Insomnia',
                'Migraine',
                'Asthma',
                'Depression',
                'Chronic Back Pain'
            ],
            'treatment_plan': [
                'Lifestyle modifications include low-sodium diet, regular exercise, weight management, stress reduction, and limiting alcohol. Medications may include ACE inhibitors, diuretics, or beta-blockers. Regular monitoring is essential.',
                'Comprehensive management includes blood glucose monitoring, medication adherence, carbohydrate counting, regular exercise, and weight management. Regular check-ups for complications screening is crucial.',
                'Treatment combines cognitive behavioral therapy, relaxation techniques, gradual exposure therapy, and sometimes medication. Lifestyle changes include regular exercise, stress management, and adequate sleep.',
                'Multimodal approach includes physical therapy, low-impact exercise, weight management, anti-inflammatory medications, and joint protection strategies. Heat and cold therapy provide symptom relief.',
                'Management includes dietary modifications, weight loss, elevation of head during sleep, and medications like proton pump inhibitors. Avoiding trigger foods and eating smaller meals helps.',
                'Sleep hygiene practices, cognitive behavioral therapy for insomnia, relaxation techniques, and sometimes short-term medication use. Addressing underlying causes is important.',
                'Acute treatment with triptans or NSAIDs, preventive medications for frequent migraines, trigger identification and avoidance, stress management, and maintaining regular sleep patterns.',
                'Controller medications for long-term management, quick-relief inhalers for acute symptoms, trigger avoidance, and asthma action plan. Regular monitoring of lung function.',
                'Combination of psychotherapy, medication, lifestyle changes, social support, and sometimes alternative therapies. Regular follow-up and monitoring for treatment response.',
                'Physical therapy, core strengthening exercises, posture correction, pain management techniques, and ergonomic modifications. Addressing underlying causes when possible.'
            ]
        }
        
        return pd.DataFrame(treatment_data)
    
    def combine_all_datasets(self) -> pd.DataFrame:
        """Combine all datasets into one comprehensive dataset"""
        comprehensive_df = self.get_comprehensive_dataset()
        symptom_df = self.get_symptom_disease_dataset()
        treatment_df = self.get_treatment_dataset()
        
        # Standardize column names
        symptom_df = symptom_df.rename(columns={
            'symptoms': 'question',
            'detailed_response': 'response'
        })
        
        treatment_df = treatment_df.rename(columns={
            'condition': 'question',
            'treatment_plan': 'response'
        })
        
        # Add question prefixes for treatment dataset
        treatment_df['question'] = 'How to treat ' + treatment_df['question'] + '?'
        
        # Combine all datasets
        combined_df = pd.concat([comprehensive_df, symptom_df[['question', 'response']], 
                                treatment_df[['question', 'response']]], ignore_index=True)
        
        return combined_df
    
    def get_pediatric_dataset(self) -> pd.DataFrame:
        """Create pediatric-specific health dataset"""
        pediatric_data = {
            'question': [
                'What are common childhood illnesses?',
                'How to manage fever in children?',
                'What causes frequent ear infections in kids?',
                'How to boost children\'s immune system?',
                'What are the signs of developmental delays?'
            ],
            'response': [
                'Common childhood illnesses include respiratory infections, ear infections, stomach bugs, and rashes. Most are viral and resolve with supportive care. Ensure proper hydration, rest, and age-appropriate medications.',
                'Fever management in children includes ensuring adequate hydration, appropriate clothing, and fever reducers like acetaminophen or ibuprofen (following age guidelines). Seek medical care for high fevers or concerning symptoms.',
                'Frequent ear infections in children can be due to anatomy, allergies, or exposure to irritants. Treatment may include antibiotics for bacterial infections, pain management, and preventive measures.',
                'Boost children\'s immunity through healthy diet, adequate sleep, regular exercise, proper hygiene, and staying up-to-date with vaccinations. Limit exposure to sick individuals when possible.',
                'Developmental delays may present as missed milestones in speech, motor skills, or social development. Early intervention is crucial - consult pediatrician for evaluation and appropriate referrals.'
            ]
        }
        
        return pd.DataFrame(pediatric_data)
    
    def get_geriatric_dataset(self) -> pd.DataFrame:
        """Create geriatric-specific health dataset"""
        geriatric_data = {
            'question': [
                'How to manage age-related health changes?',
                'What are common health issues in seniors?',
                'How to prevent falls in elderly?',
                'What causes memory problems in aging?',
                'How to maintain independence in old age?'
            ],
            'response': [
                'Age-related health changes include decreased mobility, sensory changes, and chronic conditions. Management involves regular check-ups, medication management, physical activity, proper nutrition, and social engagement.',
                'Common senior health issues include arthritis, heart disease, diabetes, osteoporosis, and cognitive changes. Regular screenings, preventive care, and chronic disease management are essential.',
                'Fall prevention includes home safety modifications, regular exercise for strength and balance, medication review, vision and hearing checks, and proper footwear.',
                'Memory problems can result from normal aging, medications, depression, or dementia. Evaluation by healthcare provider is important to determine cause and appropriate interventions.',
                'Maintaining independence involves staying physically active, managing chronic conditions, adapting home environment, utilizing community resources, and maintaining social connections.'
            ]
        }
        
        return pd.DataFrame(geriatric_data)
    
    def get_extended_dataset(self) -> pd.DataFrame:
        """Get extended dataset with all categories"""
        datasets = [
            self.get_comprehensive_dataset(),
            self.get_symptom_disease_dataset().rename(columns={'symptoms': 'question', 'detailed_response': 'response'})[['question', 'response']],
            self.get_treatment_dataset().rename(columns={'condition': 'question', 'treatment_plan': 'response'}),
            self.get_pediatric_dataset(),
            self.get_geriatric_dataset()
        ]
        
        # Modify treatment dataset questions
        datasets[2]['question'] = 'How to treat ' + datasets[2]['question'] + '?'
        
        combined_df = pd.concat(datasets, ignore_index=True)
        
        # Add metadata
        combined_df['response_length'] = combined_df['response'].str.len()
        combined_df['data_source'] = 'enhanced_comprehensive_dataset'
        
        return combined_df