# core/data_generator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

from core.knowledge_base import kb_instance

class SyntheticDataGenerator:
    """
    Генератор синтетичних даних про знос на основі фізичних моделей.
    """
    
    def __init__(self, knowledge_base=kb_instance):
        self.kb = knowledge_base
        self.rng = np.random.default_rng(42)  # Для відтворюваності результатів
        
    def generate_wear_data(self, 
                          part_id: str,
                          operating_hours: int = 1000,
                          time_step: float = 1.0,
                          noise_level: float = 0.1) -> pd.DataFrame:
        """
        Генерує синтетичні дані про знос для конкретної деталі.
        
        Args:
            part_id: ID деталі з бази знань
            operating_hours: Загальний час роботи в годинах
            time_step: Крок часу між вимірами (години)
            noise_level: Рівень шуму в даних (0-1)
            
        Returns:
            DataFrame з колонками: time, wear, load, temperature, vibration
        """
        part_info = self.kb.get_part_info(part_id)
        if not part_info:
            raise ValueError(f"Деталь з ID '{part_id}' не знайдена в базі знань")
            
        part = part_info['part']
        material = part_info['material']
        mechanism = part_info['wear_mechanism']
        
        # Генерація базових параметрів експлуатації
        time_points = np.arange(0, operating_hours + time_step, time_step)
        n_points = len(time_points)
        
        # Генерація динамічних параметрів з трендом та сезонністю
        load = self._generate_operating_parameter(n_points, base_value=50, trend=0.01, seasonality=24)
        temperature = self._generate_operating_parameter(n_points, base_value=80, trend=0.005, seasonality=12)
        vibration = self._generate_operating_parameter(n_points, base_value=2.0, trend=0.002, seasonality=6)
        
        # Розрахунок зносу на основі фізичної моделі
        wear = self._calculate_wear(mechanism.physical_model, time_points, load, 
                                  temperature, material, noise_level)
        
        # Створення DataFrame
        data = pd.DataFrame({
            'time': time_points,
            'wear': wear,
            'load': load,
            'temperature': temperature, 
            'vibration': vibration,
            'part_id': part_id,
            'material_id': material.id,
            'mechanism_id': mechanism.id
        })
        
        return data
    
    def _generate_operating_parameter(self, n_points: int, base_value: float, 
                                    trend: float, seasonality: int) -> np.ndarray:
        """Генерує реалістичний часовий ряд параметра експлуатації."""
        # Базове значення + тренд
        t = np.arange(n_points)
        parameter = base_value + trend * t
        
        # Додаємо сезонність
        if seasonality > 0:
            seasonal = 0.1 * base_value * np.sin(2 * np.pi * t / seasonality)
            parameter += seasonal
            
        # Додаємо випадкові коливання
        noise = 0.05 * base_value * self.rng.normal(0, 1, n_points)
        parameter += noise
        
        # Обрізаємо негативні значення
        parameter = np.maximum(parameter, 0.1 * base_value)
        
        return parameter
    
    def _calculate_wear(self, model_type: str, time: np.ndarray, load: np.ndarray,
                       temperature: np.ndarray, material: Material, noise_level: float) -> np.ndarray:
        """Розраховує знос на основі обраної фізичної моделі."""
        
        if model_type == 'archard':
            return self._archard_model(time, load, temperature, material, noise_level)
        elif model_type == 'fatigue':
            return self._fatigue_model(time, load, temperature, material, noise_level)
        else:
            # Модель за замовчуванням - лінійний знос
            return self._linear_model(time, load, temperature, material, noise_level)
    
    def _archard_model(self, time: np.ndarray, load: np.ndarray, temperature: np.ndarray,
                      material: Material, noise_level: float) -> np.ndarray:
        """Реалізація модифікованої моделі Арчарда для абразивного зносу."""
        # Базові параметри моделі
        K = 1e-7  # Коефіцієнт зносу
        H = material.hardness_hv
        
        # Корекція на температуру
        temp_factor = 1 + 0.01 * (temperature - 80)
        
        # Розрахунок миттєвої швидкості зносу
        wear_rate = K * load / H * temp_factor
        
        # Інтегрування для отримання кумулятивного зносу
        wear = np.cumsum(wear_rate) * (time[1] - time[0]) if len(time) > 1 else wear_rate * time[0]
        
        # Додаємо шум
        noise = noise_level * wear * self.rng.normal(0, 1, len(wear))
        wear += noise
        
        return wear
    
    def _fatigue_model(self, time: np.ndarray, load: np.ndarray, temperature: np.ndarray,
                      material: Material, noise_level: float) -> np.ndarray:
        """Реалізація моделі втомного зносу."""
        # Параметри моделі Веллера
        C = 1e-12  # Коефіцієнт втоми
        m = 3.0    # Показник степеня
        
        # Еквівалентне напруження (спрощено)
        stress = load * 0.1  # Переведення навантаження в напруження
        
        # Швидкість ушкодження
        damage_rate = C * (stress ** m)
        
        # Кумулятивне ушкодження
        damage = np.cumsum(damage_rate) * (time[1] - time[0]) if len(time) > 1 else damage_rate * time[0]
        
        # Переведення ушкодження в знос (мікрони)
        wear = damage * 1000  # Масштабування
        
        # Додаємо шум
        noise = noise_level * wear * self.rng.normal(0, 1, len(wear))
        wear += noise
        
        return wear
    
    def _linear_model(self, time: np.ndarray, load: np.ndarray, temperature: np.ndarray,
                     material: Material, noise_level: float) -> np.ndarray:
        """Проста лінійна модель зносу."""
        base_wear_rate = 0.1  # мкм/год
        load_factor = 1 + 0.01 * (load - 50)
        temp_factor = 1 + 0.005 * (temperature - 80)
        
        wear_rate = base_wear_rate * load_factor * temp_factor
        wear = wear_rate * time
        
        # Додаємо шум
        noise = noise_level * wear * self.rng.normal(0, 1, len(wear))
        wear += noise
        
        return wear
    
    def generate_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Генерує великий набір даних для навчання моделей.
        
        Args:
            n_samples: Кількість зразків для генерації
            
        Returns:
            Об'єднаний DataFrame з даними для різних деталей
        """
        all_data = []
        available_parts = self.kb.list_available_parts()
        
        if not available_parts:
            raise ValueError("Немає доступних деталей в базі знань")
        
        samples_per_part = max(1, n_samples // len(available_parts))
        
        for part_id in available_parts:
            for i in range(samples_per_part):
                # Рандомізуємо параметри для різноманітності
                operating_hours = self.rng.integers(500, 2000)
                time_step = self.rng.choice([0.5, 1.0, 2.0])
                noise_level = self.rng.uniform(0.05, 0.2)
                
                data = self.generate_wear_data(part_id, operating_hours, time_step, noise_level)
                all_data.append(data)
                
        # Об'єднуємо всі дані
        full_dataset = pd.concat(all_data, ignore_index=True)
        return full_dataset

# Глобальний екземпляр генератора
#data_generator = SyntheticDataGenerator()

if __name__ == "__main__":
    # Тестування генератора даних
    generator = SyntheticDataGenerator()
    
    # Генерація даних для поршневого кільця
    test_data = generator.generate_wear_data('piston_ring', operating_hours=100, time_step=1.0)
    print("✅ Згенеровані тестові дані:")
    print(test_data.head())
    print(f"Розмір даних: {test_data.shape}")
    
    # Генерація повного набору даних
    full_dataset = generator.generate_dataset(n_samples=100)
    print(f"\n✅ Повний набір даних: {full_dataset.shape}")
    print("Розподіл по деталях:")
    print(full_dataset['part_id'].value_counts())