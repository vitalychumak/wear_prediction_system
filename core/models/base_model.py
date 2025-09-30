# core/models/base_model.py
from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Dict, Any, Optional
import numpy as np

class BaseModel(ABC):
    """
    Абстрактний базовий клас для всіх моделей машинного навчання.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.history = None
        self.is_trained = False
        
    @abstractmethod
    def build_model(self, input_shape: tuple, **kwargs) -> tf.keras.Model:
        """Побудова архітектури моделі."""
        pass
        
    @abstractmethod
    def preprocess_data(self, X, y=None):
        """Попередня обробка даних для моделі."""
        pass
        
    def compile_model(self, optimizer='adam', loss='mse', metrics=None):
        """Компіляція моделі."""
        if self.model is None:
            raise ValueError("Модель не побудована. Викличте build_model() first.")
            
        if metrics is None:
            metrics = ['mae', 'mse']
            
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, callbacks=None, **kwargs):
        """Навчання моделі."""
        if self.model is None:
            raise ValueError("Модель не побудована.")
            
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            **kwargs
        )
        self.is_trained = True
        return self.history
        
    def predict(self, X):
        """Прогнозування."""
        if not self.is_trained:
            raise ValueError("Модель не навчена.")
        return self.model.predict(X)
        
    def evaluate(self, X_test, y_test):
        """Оцінка моделі."""
        if not self.is_trained:
            raise ValueError("Модель не навчена.")
        return self.model.evaluate(X_test, y_test)
        
    def save_model(self, filepath: str):
        """Збереження моделі."""
        if self.model is None:
            raise ValueError("Модель не існує.")
        self.model.save(filepath)
        
    def load_model(self, filepath: str):
        """Завантаження моделі."""
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        
    def get_summary(self):
        """Отримання інформації про модель."""
        if self.model is None:
            return "Модель не побудована."
        
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        return "\n".join(summary)