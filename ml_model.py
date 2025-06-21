import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

class KidneyDiseasePredictor:
    def __init__(self):
        self.model_nb = None
        self.model_knn = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = ['age', 'bp', 'bgr', 'sc', 'bu']
        self.best_k = 7
        
    def attributeSel(self, path):
        """Preprocessing data seperti kode Anda"""
        data = pd.read_csv(path, dtype=str)
        print("Kolom asli:", data.columns.tolist())
        
        # Skip jika sudah tidak ada row 1,2
        try:
            row_drop = [1, 2]
            data = data.drop(index=row_drop)
        except:
            pass
        
        data.columns = data.columns.str.strip()
        data = data[['age', 'bp', 'bgr', 'sc', 'bu', 'classification']]
        data = data.dropna(how='all')
        data = data.dropna(axis=1, how='all')
        
        # Ubah kolom ke numerik
        for col in ['age', 'bp', 'bgr', 'sc', 'bu']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Tangani missing value
        for i in data.columns:
            if data[i].isnull().any():
                if data[i].dtype in ['float64', 'int64']:
                    data[i] = data[i].fillna(round(data[i].mean()))
                else:
                    data[i] = data[i].fillna(data[i].mode()[0])
        
        return data
    
    def preprocess_data(self, data_path):
        """Preprocess data dan siapkan untuk training"""
        df = self.attributeSel(data_path)
        
        # Bersihkan classification
        df['classification'] = df['classification'].astype(str).str.strip().str.lower()
        classification_mapping = {
            'ckd': 'ckd',
            'notckd': 'notckd',
            'not ckd': 'notckd',
            'no ckd': 'notckd'
        }
        df['classification'] = df['classification'].replace(classification_mapping)
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        df['classification'] = self.label_encoder.fit_transform(df['classification'])
        
        return df
    
    def train_models(self, data_path):
        """Train kedua model"""
        print("🔄 Memulai training...")
        df = self.preprocess_data(data_path)
        
        # Pisahkan features dan target
        X = df[self.feature_columns]
        y = df['classification']
        
        print(f"📊 Data shape: {X.shape}")
        print(f"📊 Class distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature scaling untuk KNN
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Naive Bayes
        print("🧠 Training Naive Bayes...")
        self.model_nb = GaussianNB()
        self.model_nb.fit(X_train, y_train)
        
        # Train KNN
        print(f"🎯 Training KNN (K={self.best_k})...")
        self.model_knn = KNeighborsClassifier(
            n_neighbors=self.best_k,
            metric='euclidean',
            weights='uniform'
        )
        self.model_knn.fit(X_train_scaled, y_train)
        
        # Test accuracy
        y_pred_nb = self.model_nb.predict(X_test)
        y_pred_knn = self.model_knn.predict(X_test_scaled)
        
        accuracy_nb = accuracy_score(y_test, y_pred_nb)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        
        print(f"✅ Naive Bayes Accuracy: {accuracy_nb:.4f} ({accuracy_nb*100:.2f}%)")
        print(f"✅ KNN Accuracy: {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")
        
        return {
            'nb_accuracy': float(accuracy_nb),
            'knn_accuracy': float(accuracy_knn),
            'nb_accuracy_percent': float(accuracy_nb * 100),
            'knn_accuracy_percent': float(accuracy_knn * 100)
        }
    
    def save_models(self, model_dir='models'):
        """Simpan trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model_nb, os.path.join(model_dir, 'naive_bayes_model.pkl'))
        joblib.dump(self.model_knn, os.path.join(model_dir, 'knn_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
        
        print("💾 Models saved successfully!")
    
    def load_models(self, model_dir='models'):
        """Load trained models"""
        try:
            self.model_nb = joblib.load(os.path.join(model_dir, 'naive_bayes_model.pkl'))
            self.model_knn = joblib.load(os.path.join(model_dir, 'knn_model.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
            
            print("📥 Models loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_single(self, age, bp, bgr, sc, bu, method='naive_bayes'):
        """Prediksi untuk satu data input dengan pengukuran waktu"""
        # Buat DataFrame dari input
        input_data = pd.DataFrame({
            'age': [age],
            'bp': [bp],
            'bgr': [bgr],
            'sc': [sc],
            'bu': [bu]
        })
        
        # Mulai pengukuran waktu
        start_time = time.time()
        
        if method == 'naive_bayes':
            prediction = self.model_nb.predict(input_data)[0]
            probability = self.model_nb.predict_proba(input_data)[0]
            confidence = max(probability) * 100
            
        elif method == 'knn':
            input_scaled = self.scaler.transform(input_data)
            prediction = self.model_knn.predict(input_scaled)[0]
            probability = self.model_knn.predict_proba(input_scaled)[0]
            confidence = max(probability) * 100
        
        # Hitung waktu eksekusi
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Convert prediction ke label
        risk_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Tentukan risk level untuk UI
        if risk_label == 'ckd':
            risk_level = 'Tinggi'
            risk_color = 'red'
            risk_icon = 'warning'
            message = 'Hasil menunjukkan risiko TINGGI penyakit ginjal kronis.'
        else:
            risk_level = 'Rendah'
            risk_color = 'green'
            risk_icon = 'check_circle'
            message = 'Hasil menunjukkan risiko RENDAH penyakit ginjal kronis.'
        
        return {
            'risk': risk_level,
            'color': risk_color,
            'icon': risk_icon,
            'message': message,
            'probability': float(probability[1] * 100),  # Probabilitas CKD
            'confidence': float(confidence),
            'raw_prediction': risk_label,
            'execution_time': float(execution_time),
            'execution_time_ms': float(execution_time * 1000)
        }

# Instance global
predictor = KidneyDiseasePredictor()