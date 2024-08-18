import os
import pandas as pd
import joblib
from flask import Flask, request, render_template, session

app = Flask(__name__)

# Set the secret key to a complex random value for session management
app.config['SECRET_KEY'] = os.urandom(24)

# Load the trained model from disk
model = joblib.load('injury_recovery_model.pkl')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.pop('user_field', None)
    session.pop('user_mob', None)
    session.pop('sys_otp', None)
    return render_template("index.html")

@app.route('/injury_recovery')
def injury_recovery():
    return render_template("external-page/injury_recover.html")

@app.route('/injury_remedies')
def injury_remedies():
    return render_template("external-page/injuries.html")

@app.route("/submit_data", methods=["POST"])
def submit_data():
    if request.method == 'POST':
        # Print the form data for debugging
        print(request.form)
        
        injury = request.form.get('injury')
        age = int(request.form.get('age'))
        calorie = int(request.form.get('calorie'))
        gender = request.form.get('gender')
        weight = int(request.form.get('weight'))
        type1 = request.form.get('type')

        input_data = pd.DataFrame({
            'Calorie': [calorie],
            'Age': [age],
            'Weight': [weight],
            'Gender': [gender],
            'Type': [type1],
            'Injury': [injury]
        })

        # Print the input data for debugging
        print("Input data:", input_data)

        # Transform input data and make prediction
        input_data_transformed = model.named_steps['preprocessor'].transform(input_data)
        prediction = model.named_steps['model'].predict(input_data_transformed)
        ans = int(round(prediction[0]))

        # Print the prediction result for debugging
        print("Prediction result:", ans)

        return render_template("external-page/injury_recover.html", data=ans * 7)

if __name__ == '__main__':
    app.run(debug=True)
