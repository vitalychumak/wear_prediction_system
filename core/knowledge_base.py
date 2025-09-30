# core/knowledge_base.py
import pandas as pd
from typing import Dict, List, Optional
from pydantic import BaseModel

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
    physical_model: str

class MachinePart(BaseModel):
    id: str
    name: str
    typical_material_id: str
    typical_wear_mechanism_id: str

class KnowledgeBase:
    """Онтологічна база знань з CSV файлів."""
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        self.kb = knowledge_base
        self.rng = np.random.default_rng(42)

    def _load_data(self):
        """Завантажує дані з CSV-файлів."""
        try:
            # Завантаження матеріалів
            materials_df = pd.read_csv(f"{self.data_path}materials_db.csv")
            for _, row in materials_df.iterrows():
                self.materials[row['id']] = Material(**row.to_dict())
            
            # Завантаження механізмів зносу
            mechanisms_df = pd.read_csv(f"{self.data_path}wear_mechanisms.csv")
            for _, row in mechanisms_df.iterrows():
                self.wear_mechanisms[row['id']] = WearMechanism(**row.to_dict())
            
            # Завантаження деталей
            parts_df = pd.read_csv(f"{self.data_path}machine_parts.csv")
            for _, row in parts_df.iterrows():
                self.machine_parts[row['id']] = MachinePart(**row.to_dict())
            
            print("✅ База знань успішно завантажена.")
            print(f"   Матеріалів: {len(self.materials)}")
            print(f"   Механізмів зносу: {len(self.wear_mechanisms)}")
            print(f"   Деталей: {len(self.machine_parts)}")
            
        except FileNotFoundError as e:
            print(f"❌ Помилка завантаження даних: {e}")
            print("ℹ️  Створюємо тестову базу знань...")
            self._create_test_data()

    def _create_test_data(self):
        """Створює мінімальну тестову базу даних."""
        # Тестові матеріали
        self.materials['cast_iron_1'] = Material(
            id='cast_iron_1',
            name='Чавун сірий',
            hardness_hv=250,
            roughness_avg=1.2,
            young_modulus=110,
            density=7200
        )
        
        # Тестові механізми
        self.wear_mechanisms['abrasive'] = WearMechanism(
            id='abrasive',
            name='Абразивний знос',
            description='Знос, викликаний дією твердих часток.',
            physical_model='archard'
        )
        
        # Тестові деталі
        self.machine_parts['piston_ring'] = MachinePart(
            id='piston_ring',
            name='Поршневе кільце',
            typical_material_id='cast_iron_1',
            typical_wear_mechanism_id='abrasive'
        )

    def get_part_info(self, part_id: str) -> Optional[Dict]:
        """Повертає повну інформацію про деталь."""
        if part_id not in self.machine_parts:
            return None
            
        part = self.machine_parts[part_id]
        material = self.materials.get(part.typical_material_id)
        mechanism = self.wear_mechanisms.get(part.typical_wear_mechanism_id)  # ✅ ВИПРАВЛЕНО
        
        return {
            'part': part,
            'material': material,
            'wear_mechanism': mechanism
        }

    def list_available_parts(self) -> List[str]:
        """Повертає список доступних деталей."""
        return list(self.machine_parts.keys())

# Глобальний екземпляр
#kb_instance = KnowledgeBase()

if __name__ == "__main__":
    kb = KnowledgeBase()
    print("Доступні деталі:", kb.list_available_parts())
    print("Інформація про поршневе кільце:", kb.get_part_info('piston_ring'))