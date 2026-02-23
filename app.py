
from flask import Flask, render_template, request, redirect, session, send_file
import pickle
import sqlite3
from datetime import datetime
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load model
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"

# Initialize DB
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 news TEXT, result TEXT, date TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        news = request.form["news"]
        vector = tfidf.transform([news])
        prediction = model.predict(vector)[0]
        result = "Fake News" if prediction == 0 else "True News"
        date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        conn = sqlite3.connect("history.db")
        c = conn.cursor()
        c.execute("INSERT INTO history (news, result, date) VALUES (?, ?, ?)",
                  (news, result, date))
        conn.commit()
        conn.close()

        return render_template("predict.html", prediction=result)
    return render_template("predict.html")

# -------- ADMIN LOGIN --------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect("/admin")
        else:
            return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route("/admin")
def admin():
    if not session.get("admin"):
        return redirect("/login")

    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template("admin.html", data=data)

@app.route("/delete/<int:id>")
def delete(id):
    if not session.get("admin"):
        return redirect("/login")

    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return redirect("/admin")

# -------- EXPORT PDF --------

@app.route("/export")
def export():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT news, result, date FROM history")
    rows = c.fetchall()
    conn.close()

    file_path = "history.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    for row in rows:
        elements.append(Paragraph(f"News: {row[0]}", styles["Normal"]))
        elements.append(Paragraph(f"Result: {row[1]}", styles["Normal"]))
        elements.append(Paragraph(f"Date: {row[2]}", styles["Normal"]))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    return send_file(file_path, as_attachment=True)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)