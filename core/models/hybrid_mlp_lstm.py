# core/models/hybrid_mlp_lstm.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, List, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .base_model import BaseModel

class HybridMLPLSTMModel(BaseModel):
    """
    Гібридна модель MLP + LSTM для прогнозування зносу на основі часових рядів.
    """
    
    def __init__(self, name="hybrid_mlp_lstm"):
        super().__init__(name)
        self.static_scaler = StandardScaler()
        self.dynamic_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.sequence_length = 10
        self.static_features = []
        self.dynamic_features = []
        
    def build_model(self, 
                   static_input_dim: int,
                   dynamic_input_dim: int,
                   lstm_units: List[int] = [64, 32],
                   mlp_units: List[int] = [128, 64],
                   dropout_rate: float = 0.3,
                   **kwargs) -> tf.keras.Model:
        """
        Побудова гібридної архітектури MLP + LSTM.
        
        Args:
            static_input_dim: Розмірність статичних ознак
            dynamic_input_dim: Розмірність динамічних ознак
            lstm_units: Кількість нейронів у LSTM шарах
            mlp_units: Кількість нейронів у MLP шарах
            dropout_rate: Рівень dropout
        """
        
        # Вхід для статичних ознак (матеріал, тип деталі тощо)
        static_input = layers.Input(shape=(static_input_dim,), name='static_input')
        
        # Вхід для динамічних ознак (часові ряди)
        dynamic_input = layers.Input(shape=(self.sequence_length, dynamic_input_dim), 
                                   name='dynamic_input')
        
        # Обробка статичних ознак через MLP
        x_static = static_input
        for i, units in enumerate(mlp_units):
            x_static = layers.Dense(units, activation='relu', 
                                  name=f'static_dense_{i}')(x_static)
            x_static = layers.BatchNormalization(name=f'static_bn_{i}')(x_static)
            x_static = layers.Dropout(dropout_rate, name=f'static_dropout_{i}')(x_static)
        
        # Обробка динамічних ознак через LSTM
        x_dynamic = dynamic_input
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)  # Повертати послідовність для всіх, крім останнього шару
            x_dynamic = layers.LSTM(units, 
                                  return_sequences=return_sequences,
                                  name=f'lstm_{i}')(x_dynamic)
            x_dynamic = layers.BatchNormalization(name=f'dynamic_bn_{i}')(x_dynamic)
            x_dynamic = layers.Dropout(dropout_rate, name=f'dynamic_dropout_{i}')(x_dynamic)
        
        # Об'єднання статичних та динамічних ознак
        concatenated = layers.concatenate([x_static, x_dynamic], name='concatenate')
        
        # Додаткові повнозв'язні шари після об'єднання
        x = concatenated
        for i, units in enumerate([32, 16]):
            x = layers.Dense(units, activation='relu', name=f'combined_dense_{i}')(x)
            x = layers.Dropout(dropout_rate * 0.5, name=f'combined_dropout_{i}')(x)
        
        # Вихідний шар (регресія для прогнозування зносу)
        output = layers.Dense(1, activation='linear', name='output')(x)
        
        # Створення моделі
        self.model = Model(inputs=[static_input, dynamic_input], 
                          outputs=output, 
                          name=self.name)
        
        return self.model
    
    def _create_sequences(self, data: pd.DataFrame, static_features: List[str], 
                         dynamic_features: List[str], target_col: str = 'wear') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Створення послідовностей для LSTM з часових рядів.
        """
        sequences = []
        static_data = []
        targets = []
        
        # Групуємо дані по деталях (або іншому ідентифікатору серії)
        grouped = data.groupby('part_id')
        
        for part_id, group in grouped:
            group = group.sort_values('time')
            
            # Статичні ознаки (беремо перше значення для кожної деталі)
            static_values = group[static_features].iloc[0].values
            static_data.append(static_values)
            
            # Створення послідовностей для динамічних ознак
            dynamic_values = group[dynamic_features].values
            target_values = group[target_col].values
            
            for i in range(len(dynamic_values) - self.sequence_length):
                # Динамічна послідовність
                seq = dynamic_values[i:(i + self.sequence_length)]
                sequences.append(seq)
                
                # Цільове значення (значення зносу на наступному кроці)
                target = target_values[i + self.sequence_length]
                targets.append(target)
        
        return (np.array(static_data), 
                np.array(sequences), 
                np.array(targets))
    
    def preprocess_data(self, data: pd.DataFrame, 
                       static_features: List[str],
                       dynamic_features: List[str],
                       target_col: str = 'wear',
                       test_size: float = 0.2,
                       val_size: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Попередня обробка даних для гібридної моделі.
        """
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        
        # Створення послідовностей
        static_data, sequences, targets = self._create_sequences(
            data, static_features, dynamic_features, target_col
        )
        
        # Масштабування статичних ознак
        if len(static_data) > 0:
            static_data_scaled = self.static_scaler.fit_transform(static_data)
        else:
            static_data_scaled = static_data
            
        # Масштабування динамічних ознак
        original_shape = sequences.shape
        sequences_2d = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled_2d = self.dynamic_scaler.fit_transform(sequences_2d)
        sequences_scaled = sequences_scaled_2d.reshape(original_shape)
        
        # Масштабування цільової змінної
        targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # Розділення на тренувальну та тестову вибірки
        X_static_train, X_static_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
            static_data_scaled, sequences_scaled, targets_scaled, 
            test_size=test_size, random_state=42
        )
        
        # Додаткове розділення на тренувальну та валідаційну
        X_static_train, X_static_val, X_seq_train, X_seq_val, y_train, y_val = train_test_split(
            X_static_train, X_seq_train, y_train, 
            test_size=val_size/(1-test_size), random_state=42
        )
        
        return {
            'X_static_train': X_static_train, 'X_seq_train': X_seq_train, 'y_train': y_train,
            'X_static_val': X_static_val, 'X_seq_val': X_seq_val, 'y_val': y_val,
            'X_static_test': X_static_test, 'X_seq_test': X_seq_test, 'y_test': y_test
        }
    
    def train(self, processed_data: Dict, epochs: int = 100, batch_size: int = 32, **kwargs):
        """Навчання гібридної моделі."""
        if self.model is None:
            # Автоматична побудова моделі на основі розмірностей даних
            static_dim = processed_data['X_static_train'].shape[1]
            dynamic_dim = processed_data['X_seq_train'].shape[2]
            self.build_model(static_dim, dynamic_dim)
            self.compile_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10
            )
        ]
        
        history = super().train(
            [processed_data['X_static_train'], processed_data['X_seq_train']],
            processed_data['y_train'],
            [processed_data['X_static_val'], processed_data['X_seq_val']],
            processed_data['y_val'],
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            **kwargs
        )
        
        return history
    
    def predict(self, X_static: np.ndarray, X_sequence: np.ndarray) -> np.ndarray:
        """Прогнозування зносу."""
        predictions_scaled = self.model.predict([X_static, X_sequence])
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        return predictions.flatten()
    
    def evaluate_model(self, processed_data: Dict) -> Dict[str, float]:
        """Комплексна оцінка моделі."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Прогнозування на тестовій вибірці
        y_pred = self.predict(processed_data['X_static_test'], processed_data['X_seq_test'])
        y_true = self.target_scaler.inverse_transform(
            processed_data['y_test'].reshape(-1, 1)
        ).flatten()
        
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics

# Глобальний екземпляр моделі
hybrid_model = HybridMLPLSTMModel()

if __name__ == "__main__":
    # Тестування моделі
    model = HybridMLPLSTMModel()
    
    # Побудова тестової моделі
    test_model = model.build_model(static_input_dim=5, dynamic_input_dim=3)
    print("✅ Гібридна модель успішно побудована:")
    print(model.get_summary())
    
    # Тестування попередньої обробки даних
    from core.data_generator import data_generator
    
    # Генерація тестових даних
    test_data = data_generator.generate_dataset(n_samples=100)
    
    # Визначення ознак
    static_feats = ['hardness_hv', 'roughness_avg', 'young_modulus', 'density']
    dynamic_feats = ['load', 'temperature', 'vibration']
    
    print(f"\n✅ Тестові дані для обробки: {test_data.shape}")
    print("Перші 5 рядків:")
    print(test_data.head())