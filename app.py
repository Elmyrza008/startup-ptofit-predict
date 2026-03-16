import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import io
import base64
import os

app = Flask(__name__)

MODEL_PATH = 'startup_model.pkl'
DATASET_PATH = '50_Startups.csv'

model = None
df_raw = None
feature_names = []

if os.path.exists(MODEL_PATH) and os.path.exists(DATASET_PATH):
    model = joblib.load(MODEL_PATH)
    feature_names = model.feature_names_in_
    df_raw = pd.read_csv(DATASET_PATH)
    print("✅ Модель и датасет успешно загружены!")
else:
    print(f"❌ ОШИБКА: Файлы {MODEL_PATH} или {DATASET_PATH} не найдены.")

@app.route('/')
def index():
    table_data = []
    if df_raw is not None:
        table_data = df_raw.head(20).to_dict(orient='records')
    return render_template('index.html', data=table_data)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or df_raw is None:
        return jsonify({'error': 'Критическая ошибка сервера: модель или данные недоступны'}), 500
        
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Данные не получены'}), 400
            
        rd = float(data.get('rd', 0))
        adm = float(data.get('adm', 0))
        mkt = float(data.get('mkt', 0))
        state = data.get('state', '3')

        st_fl = 1.0 if state == '1' else 0.0
        st_ny = 1.0 if state == '2' else 0.0

        input_df = pd.DataFrame([[rd, adm, mkt, st_fl, st_ny]], columns=feature_names)
        prediction = model.predict(input_df)[0]

        plt.figure(figsize=(12, 6), facecolor='#0f0f0f')
        ax = plt.axes()
        ax.set_facecolor('#0f0f0f')
        
        x_raw = df_raw['R&D Spend']
        y_raw = df_raw['Profit']
        plt.scatter(x_raw, y_raw, color='#333333', s=40, alpha=0.6, label='Исторические данные (50 стартапов)')
        
        x_line = np.linspace(x_raw.min(), x_raw.max(), 100)
        line_df = pd.DataFrame({
            feature_names[0]: x_line,
            feature_names[1]: [df_raw['Administration'].mean()] * 100,
            feature_names[2]: [df_raw['Marketing Spend'].mean()] * 100,
            feature_names[3]: [0] * 100,
            feature_names[4]: [0] * 100
        })

        try:
            y_line = model.predict(line_df)
            plt.plot(x_line, y_line, color='#d4af37', linewidth=2, label='Модель Линейной Регрессии')
        except Exception as line_error:
            print(f"⚠️ Ошибка линии тренда: {line_error}")
        
        plt.scatter(rd, prediction, color='#ff4444', s=300, edgecolors='white', zorder=5, label='ВАШ ПРОГНОЗ')
        
        plt.title('Анализ корреляции: R&D Spend vs Profit', color='white', pad=25, fontsize=16)
        plt.xlabel('Инвестиции в R&D', color='white', fontsize=12)
        plt.ylabel('Прибыль (KGS)', color='white', fontsize=12)
        plt.tick_params(colors='white', labelsize=10)
        plt.grid(color='#222', linestyle='--', alpha=0.5)
        
        legend = plt.legend(facecolor='#141414', edgecolor='#333', labelcolor='white', fontsize=10, loc='upper left')
        plt.setp(legend.get_texts(), color='white')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', facecolor='#0f0f0f')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close() 

        return jsonify({
            'profit': f"{prediction:,.2f}",
            'plot': plot_url
        })
    except Exception as e:
        print(f"🔥 ОШИБКА ГРАФИКА: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)