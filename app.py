# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import sys
import os

# Додаємо шлях до кореневої папки проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Імпорт наших модулів
from core.knowledge_base import kb_instance
from core.data_generator import data_generator
from core.models.hybrid_mlp_lstm import hybrid_model
from utils.data_processor import DataProcessor

# Налаштування сторінки
st.set_page_config(
    page_title="Система прогнозування зносу деталей",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ініціалізація станів сесії
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

class WearPredictionApp:
    """Головний клас Streamlit додатку."""
    
    def __init__(self):
        self.kb = kb_instance
        self.data_generator = data_generator
        self.model = hybrid_model
        self.data_processor = DataProcessor()
        
    def run(self):
        """Запуск головного додатку."""
        st.title("🔧 Система прогнозування зносу деталей АСТ")
        st.markdown("---")
        
        # Бічна панель з навігацією
        with st.sidebar:
            st.header("Навігація")
            app_section = st.radio(
                "Оберіть розділ:",
                ["📊 Огляд системи", "🔍 База знань", "📈 Генерація даних", 
                 "🤖 Навчання моделі", "📊 Прогнозування", "📈 Аналітика"]
            )
            
            st.markdown("---")
            st.header("Статус системи")
            if st.session_state.model_trained:
                st.success("✅ Модель навчена")
            else:
                st.warning("⚠️ Модель не навчена")
                
            # Інформація про систему
            st.markdown("---")
            st.header("Інформація")
            st.info(
                "Ця система використовує гібридні моделі ML для прогнозування "
                "зносу деталей автомобільної та сільськогосподарської техніки."
            )
        
        # Маршрутизація по розділах
        if app_section == "📊 Огляд системи":
            self.show_overview()
        elif app_section == "🔍 База знань":
            self.show_knowledge_base()
        elif app_section == "📈 Генерація даних":
            self.show_data_generation()
        elif app_section == "🤖 Навчання моделі":
            self.show_model_training()
        elif app_section == "📊 Прогнозування":
            self.show_prediction()
        elif app_section == "📈 Аналітика":
            self.show_analytics()
    
    def show_overview(self):
        """Розділ огляду системи."""
        st.header("📊 Огляд системи прогнозування зносу")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Деталей у базі знань",
                value=len(self.kb.list_available_parts())
            )
            
        with col2:
            st.metric(
                label="Матеріалів у базі",
                value=len(self.kb.materials)
            )
            
        with col3:
            status = "Навчена" if st.session_state.model_trained else "Не навчена"
            st.metric(
                label="Статус моделі",
                value=status
            )
        
        st.markdown("---")
        
        # Опис системи
        st.subheader("Архітектура системи")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Компоненти системи:
            
            1. **🔍 Онтологічна база знань**
               - Деталі, матеріали, механізми зношування
               - Фізичні моделі зносу
               
            2. **📈 Генератор синтетичних даних**
               - Модель Арчарда для абразивного зносу
               - Модель втомного зносу
               - Реалістичні часові ряди
               
            3. **🤖 Гібридна ML-модель**
               - MLP для статичних ознак
               - LSTM для часових рядів
               - Трансферне навчання
               
            4. **📊 Інтерактивний інтерфейс**
               - Візуалізація результатів
               - Прогнозування в реальному часі
               - Аналітичні звіти
            """)
        
        with col2:
            # Діаграма архітектури (спрощена)
            st.image("https://via.placeholder.com/300x400/4CAF50/FFFFFF?text=System+Architecture", 
                    caption="Архітектура системи")
    
    def show_knowledge_base(self):
        """Розділ бази знань."""
        st.header("🔍 Онтологічна база знань")
        
        tab1, tab2, tab3 = st.tabs(["📋 Деталі", "🔩 Матеріали", "⚙️ Механізми зносу"])
        
        with tab1:
            st.subheader("Деталі техніки")
            parts = self.kb.list_available_parts()
            
            if parts:
                selected_part = st.selectbox("Оберіть деталь:", parts)
                part_info = self.kb.get_part_info(selected_part)
                
                if part_info:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Основна інформація:**")
                        st.write(f"🔧 Назва: {part_info['part'].name}")
                        st.write(f"🆔 ID: {part_info['part'].id}")
                        
                    with col2:
                        if part_info['material']:
                            st.write("**Матеріал:**")
                            st.write(f"📦 {part_info['material'].name}")
                            st.write(f"💎 Твердість: {part_info['material'].hardness_hv} HV")
                        
                        if part_info['wear_mechanism']:
                            st.write("**Механізм зносу:**")
                            st.write(f"⚙️ {part_info['wear_mechanism'].name}")
                            st.write(f"📝 {part_info['wear_mechanism'].description}")
            else:
                st.warning("❌ База знань порожня. Додайте дані через адмін-панель.")
        
        with tab2:
            st.subheader("Матеріали")
            if self.kb.materials:
                materials_data = []
                for material_id, material in self.kb.materials.items():
                    materials_data.append({
                        'ID': material_id,
                        'Назва': material.name,
                        'Твердість (HV)': material.hardness_hv,
                        'Шорсткість': material.roughness_avg,
                        'Модуль Юнга': material.young_modulus
                    })
                
                materials_df = pd.DataFrame(materials_data)
                st.dataframe(materials_df, use_container_width=True)
            else:
                st.info("ℹ️ Завантажте дані матеріалів у базу знань.")
        
        with tab3:
            st.subheader("Механізми зношування")
            if self.kb.wear_mechanisms:
                mechanisms_data = []
                for mechanism_id, mechanism in self.kb.wear_mechanisms.items():
                    mechanisms_data.append({
                        'ID': mechanism_id,
                        'Назва': mechanism.name,
                        'Опис': mechanism.description,
                        'Фізична модель': mechanism.physical_model
                    })
                
                mechanisms_df = pd.DataFrame(mechanisms_data)
                st.dataframe(mechanisms_df, use_container_width=True)
    
    def show_data_generation(self):
        """Розділ генерації даних."""
        st.header("📈 Генерація синтетичних даних")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Параметри генерації")
            
            # Вибір деталі
            available_parts = self.kb.list_available_parts()
            selected_part = st.selectbox("Оберіть деталь:", available_parts)
            
            # Параметри генерації
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                operating_hours = st.slider("Час роботи (год):", 100, 2000, 500)
                time_step = st.selectbox("Крок часу (год):", [0.5, 1.0, 2.0, 5.0])
            with col1_2:
                noise_level = st.slider("Рівень шуму:", 0.05, 0.5, 0.1)
                n_samples = st.number_input("Кількість зразків:", 10, 10000, 1000)
            
            # Кнопка генерації
            if st.button("🚀 Згенерувати дані", type="primary"):
                with st.spinner("Генерація даних..."):
                    try:
                        # Генерація окремих даних для перегляду
                        sample_data = self.data_generator.generate_wear_data(
                            selected_part, operating_hours, time_step, noise_level
                        )
                        
                        # Генерація повного набору для навчання
                        full_dataset = self.data_generator.generate_dataset(n_samples)
                        
                        st.session_state.training_data = full_dataset
                        st.session_state.sample_data = sample_data
                        
                        st.success(f"✅ Згенеровано {len(full_dataset)} записів!")
                        
                    except Exception as e:
                        st.error(f"❌ Помилка генерації: {e}")
        
        with col2:
            st.subheader("Попередній перегляд")
            if 'sample_data' in st.session_state:
                data = st.session_state.sample_data
                
                st.metric("Записів", len(data))
                st.metric("Колонок", len(data.columns))
                
                # Швидка статистика
                st.write("**Статистика зносу:**")
                st.write(f"Мін: {data['wear'].min():.2f} мкм")
                st.write(f"Макс: {data['wear'].max():.2f} мкм")
                st.write(f"Середнє: {data['wear'].mean():.2f} мкм")
        
        # Візуалізація згенерованих даних
        if 'sample_data' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Візуалізація даних")
            
            data = st.session_state.sample_data
            
            tab1, tab2, tab3 = st.tabs(["📈 Знос у часі", "🔥 Параметри", "📋 Таблиця"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data['time'], data['wear'], linewidth=2)
                ax.set_xlabel('Час (год)')
                ax.set_ylabel('Знос (мкм)')
                ax.set_title('Динаміка зносу у часі')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with tab2:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                # Навантаження
                axes[0, 0].plot(data['time'], data['load'])
                axes[0, 0].set_title('Навантаження')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Температура
                axes[0, 1].plot(data['time'], data['temperature'])
                axes[0, 1].set_title('Температура')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Вібрація
                axes[1, 0].plot(data['time'], data['vibration'])
                axes[1, 0].set_title('Вібрація')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Кореляція
                correlation = data[['wear', 'load', 'temperature', 'vibration']].corr()
                sns.heatmap(correlation, annot=True, ax=axes[1, 1])
                axes[1, 1].set_title('Кореляція параметрів')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab3:
                st.dataframe(data.head(100), use_container_width=True)
                
                # Експорт даних
                csv = data.to_csv(index=False)
                st.download_button(
                    label="📥 Завантажити CSV",
                    data=csv,
                    file_name=f"wear_data_{selected_part}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    def show_model_training(self):
        """Розділ навчання моделі."""
        st.header("🤖 Навчання гібридної моделі")
        
        if st.session_state.training_data is None:
            st.warning("⚠️ Спочатку згенеруйте дані у розділі 'Генерація даних'")
            return
        
        st.subheader("Параметри навчання")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Архітектура моделі:**")
            lstm_units = st.text_input("LSTM нейрони:", "64, 32")
            mlp_units = st.text_input("MLP нейрони:", "128, 64")
            dropout_rate = st.slider("Dropout rate:", 0.1, 0.5, 0.3)
        
        with col2:
            st.write("**Параметри навчання:**")
            epochs = st.number_input("Епохи:", 10, 1000, 100)
            batch_size = st.selectbox("Розмір батчу:", [16, 32, 64, 128])
            test_size = st.slider("Тестова вибірка %:", 0.1, 0.4, 0.2)
        
        # Кнопка навчання
        if st.button("🎯 Почати навчання", type="primary"):
            with st.spinner("Навчання моделі..."):
                try:
                    # Параметри попередньої обробки
                    static_features = ['hardness_hv', 'roughness_avg', 'young_modulus', 'density']
                    dynamic_features = ['load', 'temperature', 'vibration']
                    
                    # Попередня обробка даних
                    processed_data = self.model.preprocess_data(
                        st.session_state.training_data,
                        static_features=static_features,
                        dynamic_features=dynamic_features,
                        test_size=test_size
                    )
                    
                    # Побудова моделі
                    static_dim = len(static_features)
                    dynamic_dim = len(dynamic_features)
                    
                    self.model.build_model(
                        static_input_dim=static_dim,
                        dynamic_input_dim=dynamic_dim,
                        lstm_units=[int(x.strip()) for x in lstm_units.split(',')],
                        mlp_units=[int(x.strip()) for x in mlp_units.split(',')],
                        dropout_rate=dropout_rate
                    )
                    
                    self.model.compile_model()
                    
                    # Навчання моделі
                    history = self.model.train(
                        processed_data=processed_data,
                        epochs=epochs,
                        batch_size=batch_size
                    )
                    
                    st.session_state.processed_data = processed_data
                    st.session_state.model_trained = True
                    st.session_state.training_history = history.history
                    
                    st.success("✅ Модель успішно навчена!")
                    
                except Exception as e:
                    st.error(f"❌ Помилка навчання: {e}")
        
        # Візуалізація результатів навчання
        if st.session_state.model_trained:
            st.markdown("---")
            st.subheader("📊 Результати навчання")
            
            # Метрики моделі
            if st.session_state.processed_data:
                metrics = self.model.evaluate_model(st.session_state.processed_data)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with col3:
                    st.metric("R²", f"{metrics['r2']:.3f}")
                with col4:
                    st.metric("MSE", f"{metrics['mse']:.2f}")
            
            # Графіки навчання
            if 'training_history' in st.session_state:
                history = st.session_state.training_history
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Loss
                ax1.plot(history['loss'], label='Train Loss')
                if 'val_loss' in history:
                    ax1.plot(history['val_loss'], label='Val Loss')
                ax1.set_title('Функція втрат')
                ax1.set_xlabel('Епоха')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # MAE
                if 'mae' in history:
                    ax2.plot(history['mae'], label='Train MAE')
                    if 'val_mae' in history:
                        ax2.plot(history['val_mae'], label='Val MAE')
                    ax2.set_title('Середня абсолютна похибка')
                    ax2.set_xlabel('Епоха')
                    ax2.set_ylabel('MAE')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    def show_prediction(self):
        """Розділ прогнозування."""
        st.header("📊 Прогнозування зносу")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Спочатку навчіть модель у розділі 'Навчання моделі'")
            return
        
        st.subheader("Параметри для прогнозу")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Статичні параметри:**")
            hardness = st.slider("Твердість (HV):", 100, 1000, 250)
            roughness = st.slider("Шорсткість (мкм):", 0.1, 5.0, 1.2)
            young_modulus = st.slider("Модуль Юнга (GPa):", 50, 300, 110)
            density = st.slider("Густина (kg/m³):", 1000, 10000, 7200)
        
        with col2:
            st.write("**Динамічні параметри:**")
            base_load = st.slider("Базове навантаження:", 10, 200, 50)
            base_temp = st.slider("Базова температура:", 20, 150, 80)
            base_vibration = st.slider("Базова вібрація:", 0.1, 10.0, 2.0)
            prediction_hours = st.slider("Прогноз на (год):", 10, 500, 100)
        
        if st.button("🔮 Зробити прогноз", type="primary"):
            with st.spinner("Розрахунок прогнозу..."):
                try:
                    # Генерація тестових даних для прогнозу
                    static_features = np.array([[hardness, roughness, young_modulus, density]])
                    
                    # Генерація динамічних ознак
                    time_points = np.arange(prediction_hours)
                    load = base_load + 10 * np.sin(2 * np.pi * time_points / 24)
                    temperature = base_temp + 5 * np.sin(2 * np.pi * time_points / 12)
                    vibration = base_vibration + 0.5 * np.sin(2 * np.pi * time_points / 6)
                    
                    # Створення послідовностей
                    sequence_length = self.model.sequence_length
                    sequences = []
                    
                    for i in range(len(time_points) - sequence_length):
                        seq = np.column_stack([
                            load[i:i+sequence_length],
                            temperature[i:i+sequence_length],
                            vibration[i:i+sequence_length]
                        ])
                        sequences.append(seq)
                    
                    sequences = np.array(sequences)
                    
                    # Прогнозування
                    predictions = self.model.predict(static_features, sequences)
                    
                    # Збереження результатів
                    st.session_state.prediction_results = {
                        'time': time_points[sequence_length:],
                        'wear': predictions,
                        'load': load[sequence_length:],
                        'temperature': temperature[sequence_length:],
                        'vibration': vibration[sequence_length:]
                    }
                    
                    st.success("✅ Прогноз успішно розраховано!")
                    
                except Exception as e:
                    st.error(f"❌ Помилка прогнозування: {e}")
        
        # Візуалізація прогнозу
        if 'prediction_results' in st.session_state:
            st.markdown("---")
            st.subheader("📈 Результати прогнозування")
            
            results = st.session_state.prediction_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Статистика прогнозу:**")
                st.write(f"Мінімальний знос: {results['wear'].min():.2f} мкм")
                st.write(f"Максимальний знос: {results['wear'].max():.2f} мкм")
                st.write(f"Середній знос: {results['wear'].mean():.2f} мкм")
                st.write(f"Загальний знос: {results['wear'][-1]:.2f} мкм")
            
            with col2:
                # Індикатор критичності
                max_wear = results['wear'].max()
                if max_wear < 50:
                    st.success("✅ Низький рівень зносу")
                elif max_wear < 100:
                    st.warning("⚠️ Середній рівень зносу")
                else:
                    st.error("🚨 Високий рівень зносу - необхідне обслуговування!")
            
            # Графік прогнозу
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(results['time'], results['wear'], linewidth=2, color='red')
            ax.set_xlabel('Час (год)')
            ax.set_ylabel('Знос (мкм)')
            ax.set_title('Прогноз зносу')
            ax.grid(True, alpha=0.3)
            
            # Додавання ліній критичності
            ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Попередження')
            ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Критично')
            ax.legend()
            
            st.pyplot(fig)
    
    def show_analytics(self):
        """Розділ аналітики."""
        st.header("📈 Аналітика та звіти")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Для аналітики необхідно навчити модель")
            return
        
        st.subheader("Аналітичні звіти")
        
        tab1, tab2, tab3 = st.tabs(["📊 Продуктивність", "🔍 Аналіз помилок", "📈 Тренди"])
        
        with tab1:
            st.write("**Продуктивність моделі**")
            
            if st.session_state.processed_data:
                metrics = self.model.evaluate_model(st.session_state.processed_data)
                
                # Візуалізація метрик
                fig, ax = plt.subplots(figsize=(8, 4))
                metric_names = ['MAE', 'RMSE', 'R²']
                metric_values = [metrics['mae'], metrics['rmse'], metrics['r2']]
                
                bars = ax.bar(metric_names, metric_values, color=['blue', 'orange', 'green'])
                ax.set_ylabel('Значення')
                ax.set_title('Метрики якості моделі')
                
                # Додавання значень на стовпці
                for bar, value in zip(bars, metric_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                st.pyplot(fig)
        
        with tab2:
            st.write("**Аналіз помилок прогнозування**")
            
            if st.session_state.processed_data:
                # Прогнозування на тестовій вибірці для аналізу помилок
                y_pred = self.model.predict(
                    st.session_state.processed_data['X_static_test'],
                    st.session_state.processed_data['X_seq_test']
                )
                y_true = self.model.target_scaler.inverse_transform(
                    st.session_state.processed_data['y_test'].reshape(-1, 1)
                ).flatten()
                
                # Графік помилок
                errors = y_true - y_pred
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Гістограма помилок
                ax1.hist(errors, bins=50, alpha=0.7, edgecolor='black')
                ax1.set_xlabel('Похибка прогнозу')
                ax1.set_ylabel('Частота')
                ax1.set_title('Розподіл помилок прогнозування')
                ax1.grid(True, alpha=0.3)
                
                # Q-Q plot для перевірки нормальності
                from scipy import stats
                stats.probplot(errors, dist="norm", plot=ax2)
                ax2.set_title('Q-Q plot помилок')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Статистика помилок
                st.write("**Статистика помилок:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Середня помилка", f"{errors.mean():.2f}")
                with col2:
                    st.metric("Стандартне відхилення", f"{errors.std():.2f}")
                with col3:
                    st.metric("Максимальна помилка", f"{abs(errors).max():.2f}")
        
        with tab3:
            st.write("**Аналіз трендів та залежностей**")
            
            if st.session_state.training_data is not None:
                data = st.session_state.training_data
                
                # Кореляційна матриця
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                correlation = data[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Кореляція між параметрами')
                st.pyplot(fig)

def main():
    """Головна функція запуску додатку."""
    try:
        app = WearPredictionApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Помилка запуску додатку: {e}")
        st.info("ℹ️ Перевірте, чи всі модулі системи правильно ініціалізовані")

if __name__ == "__main__":
    main()