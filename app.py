from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class SpamDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.train_accuracy = 0
        self.test_accuracy = 0
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the spam dataset"""
        try:
            # Load the dataset
            df = pd.read_csv(csv_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Handle missing values
            data = df.where((pd.notnull(df)), '')
            
            # Convert labels to binary (0 for spam, 1 for ham)
            data.loc[data['label'] == 'spam', 'label'] = 0
            data.loc[data['label'] == 'ham', 'label'] = 1
            
            # Extract features and labels
            X = data['text']
            y = data['label'].astype('int')
            
            logger.info(f"Data preprocessing completed. Samples: {len(X)}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Return dummy data for demonstration if file not found
            return self.create_dummy_data()
    
    def create_dummy_data(self):
        """Create dummy data for demonstration purposes"""
        logger.info("Creating dummy data for demonstration")
        
        spam_examples = [
            "URGENT: You've won $1000000! Click here to claim your prize now!",
            "FREE MONEY! Act now, limited time offer. Call immediately!",
            "Your account will be closed! Verify your details at fake-bank.com",
            "Congratulations! You've been selected for a FREE vacation package!",
            "CLICK HERE NOW! Make money fast from home, guaranteed results!",
            "Warning: Your credit card will be charged unless you cancel now!",
            "You've inherited money from a distant relative. Send your bank details.",
            "Lose weight fast! Amazing results guaranteed or your money back!",
            "Act now! Limited time offer expires today. Don't miss out!",
            "Your computer is infected! Download our software immediately!"
        ]
        
        ham_examples = [
            "Hi, let's schedule a meeting for tomorrow at 2 PM to discuss the project.",
            "Thank you for your email. I'll get back to you by end of day.",
            "Please find the attached report for your review. Let me know if you need changes.",
            "Reminder: Team lunch is scheduled for Friday at the usual place.",
            "The quarterly results meeting has been moved to next Wednesday.",
            "I've completed the analysis you requested. The data shows positive trends.",
            "Can you please send me the updated version of the document?",
            "Great work on the presentation! The client was very impressed.",
            "The project timeline has been updated. Please check the new deadlines.",
            "Looking forward to working with you on this exciting new project."
        ]
        
        # Create labels (0 for spam, 1 for ham)
        X = spam_examples + ham_examples
        y = [0] * len(spam_examples) + [1] * len(ham_examples)
        
        return pd.Series(X), pd.Series(y)
    
    def train_model(self, csv_path=None):
        """Train the spam detection model"""
        try:
            # Load data
            if csv_path and os.path.exists(csv_path):
                X, y = self.load_and_preprocess_data(csv_path)
            else:
                X, y = self.create_dummy_data()
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=3
            )
            
            # Initialize TF-IDF Vectorizer
            self.vectorizer = TfidfVectorizer(
                min_df=1, 
                stop_words='english', 
                lowercase=True,
                max_features=5000  # Limit features for better performance
            )
            
            # Transform the training data
            X_train_features = self.vectorizer.fit_transform(X_train)
            X_test_features = self.vectorizer.transform(X_test)
            
            # Train the model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train_features, y_train)
            
            # Calculate accuracies
            train_predictions = self.model.predict(X_train_features)
            test_predictions = self.model.predict(X_test_features)
            
            self.train_accuracy = accuracy_score(y_train, train_predictions)
            self.test_accuracy = accuracy_score(y_test, test_predictions)
            
            self.is_trained = True
            
            logger.info(f"Model trained successfully!")
            logger.info(f"Training Accuracy: {self.train_accuracy:.4f}")
            logger.info(f"Test Accuracy: {self.test_accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def predict(self, email_text):
        """Predict if an email is spam or ham"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        try:
            # Transform the input text
            email_features = self.vectorizer.transform([email_text])
            
            # Make prediction
            prediction = self.model.predict(email_features)[0]
            prediction_proba = self.model.predict_proba(email_features)[0]
            
            # Calculate confidence
            confidence = max(prediction_proba) * 100
            
            # Return result
            result = {
                'is_spam': bool(prediction == 0),
                'label': 'SPAM' if prediction == 0 else 'HAM',
                'confidence': round(confidence, 2),
                'spam_probability': round(prediction_proba[0] * 100, 2),
                'ham_probability': round(prediction_proba[1] * 100, 2)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def save_model(self, model_path='spam_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Save the trained model and vectorizer"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            logger.info("Model and vectorizer saved successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path='spam_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Load a pre-trained model and vectorizer"""
        try:
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                self.is_trained = True
                logger.info("Model and vectorizer loaded successfully!")
                return True
            else:
                logger.warning("Model files not found. Training new model...")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

# Initialize the spam detector
spam_detector = SpamDetector()

# Try to load existing model, otherwise train a new one
if not spam_detector.load_model():
    # Try to train with dataset file, otherwise use dummy data
    dataset_path = 'spam_ham_dataset.csv'  # Update this path as needed
    spam_detector.train_model(dataset_path)
    spam_detector.save_model()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for spam prediction"""
    try:
        data = request.get_json()
        
        if not data or 'email_text' not in data:
            return jsonify({'error': 'No email text provided'}), 400
        
        email_text = data['email_text'].strip()
        
        if not email_text:
            return jsonify({'error': 'Email text cannot be empty'}), 400
        
        # Make prediction
        result = spam_detector.predict(email_text)
        
        return jsonify({
            'success': True,
            'result': result,
            'model_stats': {
                'train_accuracy': round(spam_detector.train_accuracy * 100, 2),
                'test_accuracy': round(spam_detector.test_accuracy * 100, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Get model information and statistics"""
    try:
        return jsonify({
            'is_trained': spam_detector.is_trained,
            'train_accuracy': round(spam_detector.train_accuracy * 100, 2),
            'test_accuracy': round(spam_detector.test_accuracy * 100, 2),
            'model_type': 'Logistic Regression',
            'vectorizer_type': 'TF-IDF'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model (useful for updating with new data)"""
    try:
        # This could accept new training data in the future
        success = spam_detector.train_model()
        
        if success:
            spam_detector.save_model()
            return jsonify({
                'success': True,
                'message': 'Model retrained successfully',
                'train_accuracy': round(spam_detector.train_accuracy * 100, 2),
                'test_accuracy': round(spam_detector.test_accuracy * 100, 2)
            })
        else:
            return jsonify({'error': 'Failed to retrain model'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Save the HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email Spam Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            text-align: center;
            padding: 40px 20px;
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: repeating-linear-gradient(
                45deg,
                transparent,
                transparent 10px,
                rgba(255, 255, 255, 0.1) 10px,
                rgba(255, 255, 255, 0.1) 20px
            );
            animation: move 20s linear infinite;
        }

        @keyframes move {
            0% { transform: translate(-50%, -50%) rotate(0deg); }
            100% { transform: translate(-50%, -50%) rotate(360deg); }
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }

        .main-content {
            padding: 40px;
        }

        .input-section {
            margin-bottom: 30px;
        }

        .input-section h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.5rem;
        }

        .email-input {
            width: 100%;
            min-height: 200px;
            padding: 20px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: all 0.3s ease;
            background: #fafafa;
        }

        .email-input:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
        }

        .button-section {
            text-align: center;
            margin: 30px 0;
        }

        .analyze-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }

        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
        }

        .analyze-btn:active {
            transform: translateY(0);
        }

        .analyze-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .analyze-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }

        .analyze-btn:hover::before {
            left: 100%;
        }

        .result-section {
            margin-top: 30px;
            padding: 25px;
            border-radius: 15px;
            min-height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            font-weight: bold;
            transition: all 0.3s ease;
            opacity: 0;
            transform: translateY(20px);
        }

        .result-section.show {
            opacity: 1;
            transform: translateY(0);
        }

        .spam-result {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            border: 3px solid #ff4757;
        }

        .ham-result {
            background: linear-gradient(45deg, #2ed573, #1dd1a1);
            color: white;
            border: 3px solid #2ed573;
        }

        .loading-result {
            background: #74b9ff;
            color: white;
        }

        .examples-section {
            margin-top: 40px;
            padding-top: 30px;
            border-top: 2px solid #e0e0e0;
        }

        .examples-section h3 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3rem;
        }

        .example-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .example-btn {
            background: white;
            border: 2px solid #667eea;
            color: #667eea;
            padding: 12px 20px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
        }

        .example-btn:hover {
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }

        .stats-section {
            background: linear-gradient(45deg, #f8f9fa, #e9ecef);
            padding: 25px;
            border-radius: 15px;
            margin-top: 30px;
            text-align: center;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .stat-item {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }

        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }

        .stat-label {
            color: #666;
            margin-top: 5px;
        }

        .error-message {
            background: #ff7675;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            text-align: center;
        }

        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 15px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .main-content {
                padding: 20px;
            }

            .example-buttons {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚫 Email Spam Detector</h1>
            <p>AI-powered email classification with Machine Learning</p>
        </div>

        <div class="main-content">
            <div class="input-section">
                <h2>📧 Enter Email Content</h2>
                <textarea 
                    id="emailInput" 
                    class="email-input" 
                    placeholder="Paste your email content here... Include subject line and full message body for best results."
                ></textarea>
            </div>

            <div class="button-section">
                <button id="analyzeBtn" class="analyze-btn">
                    🔍 Analyze Email
                </button>
            </div>

            <div id="resultSection" class="result-section">
                Ready to analyze your email!
            </div>

            <div class="examples-section">
                <h3>📝 Try These Examples</h3>
                <div class="example-buttons">
                    <button class="example-btn" onclick="loadExample('spam1')">
                        🎁 Prize Winner Spam
                    </button>
                    <button class="example-btn" onclick="loadExample('spam2')">
                        💰 Financial Scam
                    </button>
                    <button class="example-btn" onclick="loadExample('ham1')">
                        💼 Business Meeting
                    </button>
                    <button class="example-btn" onclick="loadExample('ham2')">
                        📋 Project Update
                    </button>
                </div>
            </div>

            <div class="stats-section">
                <h3>📊 Model Performance</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-number" id="trainAccuracy">Loading...</div>
                        <div class="stat-label">Training Accuracy</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number" id="testAccuracy">Loading...</div>
                        <div class="stat-label">Test Accuracy</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">TF-IDF</div>
                        <div class="stat-label">Feature Extraction</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">Logistic</div>
                        <div class="stat-label">Regression Model</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Example emails
        const examples = {
            spam1: `Subject: Congratulations! You've Won a Free Vacation!

Dear Valued Customer,

We are pleased to announce that you have been randomly selected to receive a FREE vacation package to the exotic island of Paradise Beach!

To claim your prize, simply click on the link below and follow the instructions:
www.fake-vacation-prize.com/claim-now

Don't miss out on this incredible opportunity!
This offer is only valid for the next 24 hours.

Sincerely,
The Prize Redemption Team`,

            spam2: `Subject: URGENT: Your Account Will Be Closed

IMMEDIATE ACTION REQUIRED!!!

Your bank account will be suspended in 24 hours due to suspicious activity. 

Click here NOW to verify your identity and prevent account closure:
www.fake-bank-security.com/verify

Enter your:
- Social Security Number
- Account Number
- PIN
- Mother's Maiden Name

ACT FAST! Time is running out!

Security Department
First National Bank`,

            ham1: `Subject: Meeting Minutes - July 26th Project Update

Hi team,

Please find attached the minutes from yesterday's project meeting. We discussed the following key points:

• Q3 deliverables and timeline adjustments
• Budget allocation for the new marketing campaign
• Upcoming client presentation scheduled for next week
• Resource allocation for the development team

Next meeting is scheduled for August 2nd at 2:00 PM in Conference Room B.

Let me know if you have any questions or need clarification on any items.

Best regards,
Sarah Johnson
Project Manager`,

            ham2: `Subject: Family Dinner This Sunday

Hi everyone,

Just a reminder that we're having our monthly family dinner this Sunday at 6 PM at Mom and Dad's house.

Please let me know if you can make it so we can plan accordingly. Also, if anyone wants to bring a dish to share, that would be great!

Looking forward to seeing everyone.

Love,
Jennifer

P.S. - Don't forget it's Dad's birthday next week, so we should discuss gift ideas.`
        };

        function loadExample(exampleKey) {
            document.getElementById('emailInput').value = examples[exampleKey];
        }

        async function analyzeEmail() {
            const emailText = document.getElementById('emailInput').value.trim();
            const resultSection = document.getElementById('resultSection');
            const analyzeBtn = document.getElementById('analyzeBtn');

            if (!emailText) {
                showError('Please enter some email content to analyze.');
                return;
            }

            // Show loading state
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = '🔄 Analyzing...';
            resultSection.textContent = '🔄 Processing email with AI model...';
            resultSection.className = 'result-section show loading-result';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ email_text: emailText })
                });

                const data = await response.json();

                if (data.success) {
                    displayResult(data.result);
                    updateModelStats(data.model_stats);
                } else {
                    showError(data.error || 'Failed to analyze email');
                }
            } catch (error) {
                showError('Network error. Please try again.');
                console.error('Error:', error);
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = '🔍 Analyze Email';
            }
        }

        function displayResult(result) {
            const resultSection = document.getElementById('resultSection');
            
            if (result.is_spam) {
                resultSection.innerHTML = `
                    <div>
                        <span style="font-size: 2rem;">🚨</span><br>
                        <strong>SPAM DETECTED</strong><br>
                        <span style="font-size: 0.9rem; opacity: 0.9;">
                            Confidence: ${result.confidence}%<br>
                            Spam Probability: ${result.spam_probability}%
                        </span>
                    </div>
                `;
                resultSection.className = 'result-section show spam-result';
            } else {
                resultSection.innerHTML = `
                    <div>
                        <span style="font-size: 2rem;">✅</span><br>
                        <strong>LEGITIMATE EMAIL (HAM)</strong><br>
                        <span style="font-size: 0.9rem; opacity: 0.9;">
                            Confidence: ${result.confidence}%<br>
                            Ham Probability: ${result.ham_probability}%
                        </span>
                    </div>
                `;
                resultSection.className = 'result-section show ham-result';
            }
        }

        function showError(message) {
            const resultSection = document.getElementById('resultSection');
            resultSection.innerHTML = `<div class="error-message">❌ ${message}</div>`;
            resultSection.className = 'result-section show';
            resultSection.style.background = '#ff7675';
            resultSection.style.color = 'white';
        }

        function updateModelStats(stats) {
            if (stats) {
                document.getElementById('trainAccuracy').textContent = `${stats.train_accuracy}%`;
                document.getElementById('testAccuracy').textContent = `${stats.test_accuracy}%`;
            }
        }

        async function loadModelInfo() {
            try {
                const response = await fetch('/model_info');
                const data = await response.json();
                updateModelStats(data);
            } catch (error) {
                console.error('Error loading model info:', error);
            }
        }

        // Event listeners
        document.getElementById('analyzeBtn').addEventListener('click', analyzeEmail);

        document.getElementById('emailInput').addEventListener('keypress', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                analyzeEmail();
            }
        });

        // Load model info on page load
        window.addEventListener('load', function() {
            loadModelInfo();
            
            setTimeout(() => {
                document.querySelector('.container').style.opacity = '1';
                document.querySelector('.container').style.transform = 'translateY(0)';
            }, 100);
        });

        // Initial styles for smooth loading
        document.querySelector('.container').style.opacity = '0';
        document.querySelector('.container').style.transform = 'translateY(20px)';
        document.querySelector('.container').style.transition = 'all 0.6s ease';
    </script>
</body>
</html>'''
    
    # Write the HTML template
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("🚀 Starting Flask Spam Detector App...")
    print("📧 Your spam detection model is ready!")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("\n📊 Model Features:")
    print("   • Real-time spam detection")
    print("   • TF-IDF feature extraction")
    print("   • Logistic regression classification")
    print("   • Interactive web interface")
    print("   • Model performance metrics")
    
    app.run(debug=True, host='0.0.0.0', port=5000)