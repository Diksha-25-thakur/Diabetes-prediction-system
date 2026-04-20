from flask import Flask, render_template, request, redirect
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

# Save data
def save_data(values, result, prob):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS records (
        pregnancies REAL,
        glucose REAL,
        bp REAL,
        skin REAL,
        insulin REAL,
        bmi REAL,
        dpf REAL,
        age REAL,
        result TEXT,
        probability REAL
    )""")

    c.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)",
              (*values, result, prob))

    conn.commit()
    conn.close()


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict')
def predict():
    return render_template("predict.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/history')
def history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    # 🔥 ADD THIS PART (IMPORTANT)
    c.execute("""CREATE TABLE IF NOT EXISTS records (
        pregnancies REAL,
        glucose REAL,
        bp REAL,
        skin REAL,
        insulin REAL,
        bmi REAL,
        dpf REAL,
        age REAL,
        result TEXT,
        probability REAL
    )""")

    data = c.execute("SELECT * FROM records").fetchall()
    conn.close()

    return render_template("history.html", data=data)

@app.route('/result', methods=['POST'])
def result():
    values = [float(x) for x in request.form.values()]
    data = np.array(values).reshape(1, -1)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        result = "Diabetic ❌"
        color = "red"
    else:
        result = "Not Diabetic ✅"
        color = "green"

    save_data(values, result, probability)

    return render_template("predict.html",
                           result=result,
                           prob=round(probability * 100, 2),
                           color=color)


if __name__ == '__main__':
    app.run(debug=True)