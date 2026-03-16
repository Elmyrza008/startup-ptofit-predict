import joblib
import pandas as pd
import numpy as np

try:
    model = joblib.load('startup_model.pkl')
    feature_names = model.feature_names_in_
except Exception as e:
    print(f"Ошибка: Не удалось загрузить модель. {e}")
    exit()

print("\n=== ВВОД ДАННЫХ ДЛЯ ПРОГНОЗА ===")


try:
    rd = float(input("R&D Spend: "))
    adm = float(input("Administration: "))
    mkt = float(input("Marketing Spend: "))
    
    print("\nШтат: 1 - Florida, 2 - New York, 3 - California/Other")
    st_choice = input("Выбор: ")

    st_fl = 1.0 if st_choice == '1' else 0.0
    st_ny = 1.0 if st_choice == '2' else 0.0

    input_df = pd.DataFrame([[rd, adm, mkt, st_fl, st_ny]], columns=feature_names)

    prediction = model.predict(input_df)[0]

    print("\n" + "="*40)
    print("       РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ")
    print("="*40)
    
    summary = input_df.copy()
    summary['PREDICTED PROFIT'] = prediction
    print(summary.T) 
    
    print("="*40)
    print(f"ИТОГОВАЯ ПРИБЫЛЬ: {prediction:,.2f} KGS")
    print("="*40)

except ValueError:
    print("Ошибка: Вводите только числа!")