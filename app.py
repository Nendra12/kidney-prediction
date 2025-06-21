from flask import Flask, render_template, request, jsonify
from ml_model import predictor
from datetime import datetime
import os

app = Flask(__name__)

# Variable untuk tracking training status
models_loaded = False

def initialize_models():
    """Initialize models saat startup"""
    global models_loaded
    
    # Coba load model yang sudah ada
    if predictor.load_models():
        models_loaded = True
        print("✅ Models loaded from saved files")
    else:
        # Jika tidak ada, train model baru
        print("🔄 Training new models...")
        data_path = 'data/kidney_disease.csv'
        
        if os.path.exists(data_path):
            try:
                results = predictor.train_models(data_path)
                predictor.save_models()
                models_loaded = True
                print("✅ Models trained and saved successfully")
                print(f"📊 Naive Bayes Accuracy: {results['nb_accuracy_percent']:.2f}%")
                print(f"📊 KNN Accuracy: {results['knn_accuracy_percent']:.2f}%")
            except Exception as e:
                print(f"❌ Error training models: {e}")
                models_loaded = False
        else:
            print(f"❌ Data file not found: {data_path}")
            models_loaded = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global models_loaded
    
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Please check the console for training status.'}), 500
    
    try:
        # Ambil data dari request
        data = request.get_json()
        
        age = float(data['age'])
        bp = float(data['bp'])
        bgr = float(data['bgr'])
        sc = float(data['sc'])
        bu = float(data['bu'])
        method = data['method']
        
        # Validasi input
        if not all([age > 0, bp > 0, bgr > 0, sc > 0, bu > 0]):
            return jsonify({'error': 'Semua nilai harus lebih besar dari 0'}), 400
        
        print(f"🔍 Predicting with {method}: age={age}, bp={bp}, bgr={bgr}, sc={sc}, bu={bu}")
        
        # Prediksi
        result = predictor.predict_single(age, bp, bgr, sc, bu, method)
        
        print(f"✅ Prediction result: {result['risk']} ({result['probability']:.1f}%)")
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/compare', methods=['POST'])
def compare_algorithms():
    """Endpoint khusus untuk membandingkan kedua algoritma"""
    global models_loaded
    
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        # Ambil data dari request
        data = request.get_json()
        
        age = float(data['age'])
        bp = float(data['bp'])
        bgr = float(data['bgr'])
        sc = float(data['sc'])
        bu = float(data['bu'])
        
        # Validasi input
        if not all([age > 0, bp > 0, bgr > 0, sc > 0, bu > 0]):
            return jsonify({'error': 'Semua nilai harus lebih besar dari 0'}), 400
        
        print(f"🔍 Comparing algorithms for: age={age}, bp={bp}, bgr={bgr}, sc={sc}, bu={bu}")
        
        # Prediksi dengan Naive Bayes
        result_nb = predictor.predict_single(age, bp, bgr, sc, bu, 'naive_bayes')
        
        # Prediksi dengan KNN
        result_knn = predictor.predict_single(age, bp, bgr, sc, bu, 'knn')
        
        # Analisis perbandingan
        agreement = result_nb['raw_prediction'] == result_knn['raw_prediction']
        confidence_diff = abs(result_nb['confidence'] - result_knn['confidence'])
        probability_diff = abs(result_nb['probability'] - result_knn['probability'])
        
        # Tentukan algoritma mana yang lebih yakin
        more_confident = 'naive_bayes' if result_nb['confidence'] > result_knn['confidence'] else 'knn'
        
        # Tentukan algoritma mana yang lebih cepat
        nb_faster = result_nb['execution_time'] < result_knn['execution_time']
        
        # Generate rekomendasi berdasarkan agreement
        if agreement:
            if result_nb['raw_prediction'] == 'ckd':
                recommendation = "⚠️ KEDUA algoritma menunjukkan risiko TINGGI. Segera konsultasi dengan dokter!"
                recommendation_color = "red"
                recommendation_icon = "warning"
            else:
                recommendation = "✅ KEDUA algoritma menunjukkan risiko RENDAH. Pertahankan gaya hidup sehat!"
                recommendation_color = "green"
                recommendation_icon = "check_circle"
        else:
            recommendation = "🤔 Algoritma memberikan hasil BERBEDA. Disarankan konsultasi medis untuk kepastian."
            recommendation_color = "orange"
            recommendation_icon = "help"
        
        # Generate timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        comparison_result = {
            'naive_bayes': result_nb,
            'knn': result_knn,
            'comparison': {
                'agreement': agreement,
                'confidence_difference': round(confidence_diff, 2),
                'probability_difference': round(probability_diff, 2),
                'more_confident_algorithm': more_confident,
                'recommendation': recommendation,
                'recommendation_color': recommendation_color,
                'recommendation_icon': recommendation_icon,
                'speed_comparison': {
                    'nb_faster': nb_faster,
                    'speed_diff_ms': round(abs(result_nb['execution_time_ms'] - result_knn['execution_time_ms']), 2)
                }
            },
            'metadata': {
                'timestamp': current_time,
                'user': "Danendra1204",
                'input_data': {
                    'age': age,
                    'bp': bp,
                    'bgr': bgr,
                    'sc': sc,
                    'bu': bu
                }
            }
        }
        
        print(f"✅ Comparison completed - Agreement: {agreement}")
        
        return jsonify(comparison_result)
        
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return jsonify({'error': f'Comparison error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded
    })

if __name__ == '__main__':
    print("🚀 Starting Flask application...")
    initialize_models()
    print("🌐 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)