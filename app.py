from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your trained model
with open('model3.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your StandardScaler
with open('sc.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Load the input data from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Preprocess the inputs
        inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        inputs_scaled = scaler.transform(inputs)

        # Make prediction
        prediction = model.predict(inputs_scaled)[0]

        # Pass all the variables to the result template
        return render_template('result.html', 
                               prediction=prediction, 
                               age=age, 
                               sex=sex, 
                               cp=cp, 
                               trestbps=trestbps, 
                               chol=chol, 
                               fbs=fbs, 
                               restecg=restecg, 
                               thalach=thalach, 
                               exang=exang, 
                               oldpeak=oldpeak, 
                               slope=slope, 
                               ca=ca, 
                               thal=thal)


if __name__ == '__main__':
    app.run(debug=True)
