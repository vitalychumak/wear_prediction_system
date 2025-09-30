# utils/data_processor.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class DataProcessor:
    """Допоміжний клас для обробки даних."""
    
    def __init__(self):
        pass
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Очищення даних від аномалій та заповнення пропусків."""
        # Видалення дублікатів
        data = data.drop_duplicates()
        
        # Заповнення пропусків
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        return data
    
    def detect_anomalies(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Виявлення аномалій у даних."""
        if method == 'iqr':
            return self._iqr_method(data)
        elif method == 'zscore':
            return self._zscore_method(data)
        else:
            return data
    
    def _iqr_method(self, data: pd.DataFrame) -> pd.DataFrame:
        """Виявлення аномалій методом міжквартильного розмаху."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        clean_data = data.copy()
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Заміна аномалій медіаною
            mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            clean_data.loc[mask, col] = data[col].median()
        
        return clean_data
    
    def _zscore_method(self, data: pd.DataFrame) -> pd.DataFrame:
        """Виявлення аномалій методом Z-score."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        clean_data = data.copy()
        
        for col in numeric_cols:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            mask = z_scores > 3  # Поріг 3 стандартних відхилень
            
            # Заміна аномалій медіаною
            clean_data.loc[mask, col] = data[col].median()
        
        return clean_data
    
    def calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Розрахунок статистичних показників даних."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median(),
                'q25': data[col].quantile(0.25),
                'q75': data[col].quantile(0.75)
            }
        
        return stats