# core/knowledge_base.py
import pandas as pd
from typing import Dict, List, Optional
from pydantic import BaseModel  # Для строгої типізації даних

# Моделі даних для онтології (Pydantic моделі)
class Material(BaseModel):
    id: str
    name: str
    hardness_hv: float
    roughness_avg: float
    young_modulus: float
    density: float

class WearMechanism(BaseModel):
    id: str
    name: str
    description: str
    physical_model: str  # Назва відповідної фізичної моделі

class MachinePart(BaseModel):
    id: str
    name: str
    typical_material_id: str
    typical_wear_mechanism_id: str

class KnowledgeBase:
    """
    Проста онтологічно-орієнтована база знань, що завантажує дані з CSV-файлів.
    """
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        self.materials: Dict[str, Material] = {}
        self.wear_mechanisms: Dict[str, WearMechanism] = {}
        self.machine_parts: Dict[str, MachinePart] = {}
        
        self._load_data()

    def _load_data(self):
        """Завантажує дані з CSV-файлів у словники."""
        try:
            # Завантаження матеріалів
            materials_df = pd.read_csv(f"{self.data_path}materials_db.csv")
            for _, row in materials_df.iterrows():
                self.materials[row['id']] = Material(**row.to_dict())
                
            # TODO: Додати завантаження wear_mechanisms.csv та machine_parts.csv
            # Для прикладу, створимо тестові дані вручну
            self.wear_mechanisms['abrasive'] = WearMechanism(
                id='abrasive',
                name='Абразивний знос',
                description='Знос, викликаний дією твердих часток.',
                physical_model='archard'
            )
            self.wear_mechanisms['fatigue'] = WearMechanism(
                id='fatigue',
                name='Втомний знос',
                description='Знос через циклічне навантаження.',
                physical_model='fatigue'
            )
            
            self.machine_parts['piston_ring'] = MachinePart(
                id='piston_ring',
                name='Поршневе кільце',
                typical_material_id='cast_iron_1',
                typical_wear_mechanism_id='abrasive'
            )
            
            print("✅ База знань успішно завантажена.")
            
        except FileNotFoundError as e:
            print(f"❌ Помилка завантаження даних: {e}")
            print("ℹ️  Створюємо мінімальну тестову базу знань...")

    def get_part_info(self, part_id: str) -> Optional[Dict]:
        """Повертає повну інформацію про деталь за її ID."""
        if part_id not in self.machine_parts:
            return None
            
        part = self.machine_parts[part_id]
        material = self.materials.get(part.typical_material_id)
        mechanism = self.materials.get(part.typical_wear_mechanism_id)
        
        return {
            'part': part,
            'material': material,
            'wear_mechanism': mechanism
        }

    def list_available_parts(self) -> List[str]:
        """Повертає список доступних деталей."""
        return list(self.machine_parts.keys())

# Синглтон-екземпляр для використання по всьому додатку
kb_instance = KnowledgeBase()

if __name__ == "__main__":
    # Тестування модуля
    kb = KnowledgeBase()
    print("Доступні деталі:", kb.list_available_parts())
    print("Інформація про поршневе кільце:", kb.get_part_info('piston_ring'))