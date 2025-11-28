# ML Model Deployment using Flask and FastAPI (Penguin Species Prediction)

A small learning project showing how to train a machine-learning model on the **Seaborn Penguins dataset** and deploy it using:

* **Flask**
* **FastAPI (basic API)**
* **FastAPI with HTML frontend**

---

## 📁 Project Structure

```
|-- templates/
|     └── index.html          # Frontend for Flask / FastAPI HTML version
|-- app.py                    # Flask app deployment
|-- main.py                   # FastAPI API-only version
|-- main2.py                  # FastAPI with HTML frontend
|-- penguin.ipynb             # Model training + saving workflow
|-- requirements.txt          # Required Python packages
|-- .gitignore
```

---

## 🧠 What This Project Does

1. Loads and cleans the **Penguins dataset** (from Seaborn).
2. Trains a simple ML model (classification).
3. Saves the model using pickle.
4. Deploys the model with:

   * Flask (form-based prediction)
   * FastAPI (JSON API)
   * FastAPI + HTML (hybrid UI)

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run Flask app

```
python app.py
```

Open in browser:
`http://127.0.0.1:5000/`

### 3. Run FastAPI API

```
python main.py
```

Docs available at:
`http://127.0.0.1:8000/docs`

### 4. FastAPI with HTML UI

```
python main2.py
```

Access in browser:
`http://127.0.0.1:8000/`

---

## 📒 Notebook (Training)

`penguin.ipynb` covers:

* Loading the Seaborn dataset
* Feature prep
* Model training
* Saving encoders + model using pickle

---

