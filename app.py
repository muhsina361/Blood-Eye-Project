import os
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import uuid

# Import updated feature extraction functions
from feature_Code_fundus import extract_fundus_features
from feature_Code_scelera import extract_sclera_features

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['DATABASE'] = 'database.db'

# Updated feature list: 5 from fundus + 5 from sclera
ALL_FEATURES = [
    'fundus_cnn_pca1', 'fundus_AVR', 'fundus_vessel_redness',
    'fundus_tortuosity', 'fundus_vessel_density',
    'sclera_cnn_pca1', 'sclera_AVR', 'sclera_mean_hue',
    'sclera_redness', 'sclera_perivascular_contrast'
]

db_setup_done = False

def init_db():
    with sqlite3.connect(app.config['DATABASE']) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

@app.before_request
def setup_app():
    global db_setup_done
    if not db_setup_done:
        init_db()
        if not all(os.path.exists(f) for f in ['trained_model.h5', 'scaler.pkl', 'encoder.pkl']):
            print("Model files are missing. Running automatic training...")
            try:
                train_and_evaluate_model('eye_dataset2.csv')
                print("Automatic training complete. Model files are now available.")
            except FileNotFoundError:
                print("Error: 'eye_dataset2.csv' not found.")
            except Exception as e:
                print(f"An error occurred during automatic training: {e}")
        db_setup_done = True

def login_required(func):
    def wrapper(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            with sqlite3.connect(app.config['DATABASE']) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already exists.')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect(app.config['DATABASE']) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
            user = cursor.fetchone()
        if user:
            session['username'] = user[1]
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid credentials.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/working')
def working():
    return render_template('working.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/fundus_upload', methods=['GET', 'POST'])
@login_required
def fundus_upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = secure_filename(file.filename)
            unique_filename = str(uuid.uuid4()) + "_" + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            try:
                features, processed_image_filename = extract_fundus_features(filepath)
                features = {key: float(value) for key, value in features.items()}
                session['fundus_features'] = features
                
                processed_image_path = url_for('static', filename=os.path.join('uploads', processed_image_filename))
                
                return jsonify({
                    'success': True,
                    'features': features,
                    'processed_image_path': processed_image_path
                })
            except Exception as e:
                return jsonify({'error': str(e)})
    return render_template('fundus_upload.html')

@app.route('/sclera_upload', methods=['GET', 'POST'])
@login_required
def sclera_upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = secure_filename(file.filename)
            unique_filename = str(uuid.uuid4()) + "_" + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            try:
                features = extract_sclera_features(filepath)
                features = {key: float(value) for key, value in features.items()}
                session['sclera_features'] = features
                
                processed_image_path = url_for('static', filename=os.path.join('uploads', unique_filename))
                
                return jsonify({
                    'success': True,
                    'features': features,
                    'processed_image_path': processed_image_path
                })
            except Exception as e:
                return jsonify({'error': str(e)})
    return render_template('sclera_upload.html')

@app.route('/results')
@login_required
def results():
    fundus_features = session.get('fundus_features', None)
    sclera_features = session.get('sclera_features', None)

    if not fundus_features or not sclera_features:
        return render_template('results.html', error='Please upload both fundus and sclera images first.')

    # Combine selected features
    all_features_dict = {**fundus_features, **sclera_features}

    # Check for missing features
    for f in ALL_FEATURES:
        if f not in all_features_dict:
            return render_template('results.html', error=f"Missing feature: {f}")

    input_data = [all_features_dict[f] for f in ALL_FEATURES]
    input_data = np.array(input_data).reshape(1, -1)

    try:
        if not os.path.exists('trained_model.h5') or not os.path.exists('scaler.pkl') or not os.path.exists('encoder.pkl'):
            return render_template('results.html', error="Model files missing. Please train the model.")

        model = load_model('trained_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)
        predicted_blood_group = encoder.inverse_transform(predicted_label)[0]
    except Exception as e:
        predicted_blood_group = f"Error in prediction: {e}"

    return render_template('results.html', 
                           fundus_features=fundus_features, 
                           sclera_features=sclera_features, 
                           predicted_blood_group=predicted_blood_group)

@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html')

@app.route('/train', methods=['GET', 'POST'])
@login_required
def train_model():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            return redirect(request.url)
        file = request.files['dataset']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                train_and_evaluate_model(filepath)
                return jsonify({'message': 'Training and evaluation complete. Check the Accuracy page for results.'})
            except Exception as e:
                return jsonify({'error': str(e)})
    return render_template('train.html')

def train_and_evaluate_model(dataset_path):
    df = pd.read_csv(dataset_path)
    features_to_use = ['cnn_pca1', 'AVR', 'vessel_redness', 'sclera_mean_hue', 'AV_sat_diff', 'tortuosity', 'sclera_redness', 'vessel_density', 'perivascular_contrast', 'pulse_std']
    target = 'blood_group'
    df_clean = df.dropna(subset=features_to_use + [target])
    X = df_clean[features_to_use]
    y = df_clean[target]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)
    y_test_labels = np.argmax(y_test, axis=1)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(y_categorical.shape[1], activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        epochs=150,
                        batch_size=16,
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=1)

    model.save('trained_model1.h5')
    joblib.dump(scaler, 'scaler1.pkl')
    joblib.dump(le, 'encoder1.pkl')

    # Accuracy & Loss plots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'accuracy_loss_graphs.png'))
    plt.close(fig)

    # Confusion Matrix
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    class_names = le.classes_
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'confusion_matrix.png'))
    plt.close()

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    app.run(debug=True, use_reloader=False)
