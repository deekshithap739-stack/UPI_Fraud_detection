from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
from jinja2 import ChoiceLoader, FileSystemLoader

# ✅ Load templates from both root and pages/
app = Flask(__name__, static_folder='pages/css')

app.jinja_loader = ChoiceLoader([
    FileSystemLoader('.'),
    FileSystemLoader('pages')
])

# ✅ Load model and encoders
model = joblib.load('model/model.pkl')
le_sender = joblib.load('model/le_sender.pkl')
le_receiver = joblib.load('model/le_receiver.pkl')

# ✅ Static page routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/faq')
def faq_page():
    return render_template('faq.html')

@app.route('/feedback')
def feedback_page():
    return render_template('feedback.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/model')
def model_page():
    return render_template('model.html')

@app.route('/admin')
def admin_page():
    return render_template('admin.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')
@app.route('/about')
def about_page():
    return render_template('about.html')


@app.route('/privacy')
def privacy_page():
    return render_template('privacy.html')

@app.route('/terms')
def terms_page():
    return render_template('terms.html')

@app.route('/policy')
def policy_page():
    return render_template('policy.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/dataset')
def dataset_page():
    return render_template('dataset.html')

@app.route('/sample')
def sample_dataset():
    return send_file('sample.csv', as_attachment=True)

# ✅ Upload logic
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        df.fillna({'SenderID': 'unknown', 'ReceiverID': 'unknown', 'Amount': 0}, inplace=True)
        df['sender'] = df['SenderID'].apply(lambda x: le_sender.transform([x])[0] if x in le_sender.classes_ else 0)
        df['receiver'] = df['ReceiverID'].apply(lambda x: le_receiver.transform([x])[0] if x in le_receiver.classes_ else 0)

        predictions = model.predict(df[['Amount', 'sender', 'receiver']])
        df['prediction'] = ['Fraudulent' if p == 1 else 'Legitimate' for p in predictions]

        return df.to_html(classes='table table-bordered')

    except Exception as e:
        return f"❌ Error during upload: {str(e)}"

# ✅ Prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    try:
        amount = float(request.form['amount'])
        sender = request.form['sender']
        receiver = request.form['receiver']

        # Handle unseen or digit-only IDs
        sender_encoded = le_sender.transform([sender])[0] if sender in le_sender.classes_ else 0
        receiver_encoded = le_receiver.transform([receiver])[0] if receiver in le_receiver.classes_ else 0

        features = [[amount, sender_encoded, receiver_encoded]]
        prediction = model.predict(features)[0]
        result = "Fraudulent" if prediction == 1 else "Legitimate"

        return render_template('predict.html', result=result)

    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"

# ✅ Run the app
if __name__ == '__main__':
    app.run(debug=True)

