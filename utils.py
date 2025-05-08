import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    """
    Charger les données de consommation d'énergie depuis un fichier CSV.
    
    Arguments :
        file_path (str) : Chemin vers le fichier CSV
        
    Retourne :
        DataFrame : Données chargées
    """
    # Charger les données
    df = pd.read_csv(file_path)
    
    # Convertir la colonne timestamp en datetime si c'est une chaîne de caractères
    # Vérifier si nous avons la colonne standard ou la colonne personnalisée
    if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'datetime_utc' in df.columns and df['datetime_utc'].dtype == 'object':
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        # Ajouter une colonne timestamp pour la compatibilité avec le reste du code
        df['timestamp'] = df['datetime_utc']
        
    return df

def preprocess_data(df):
    """
    Prétraiter les données de consommation d'énergie.
    
    Arguments :
        df (DataFrame) : Données brutes
        
    Retourne :
        DataFrame : Données prétraitées
    """
    # Créer une copie pour éviter de modifier les données originales
    processed_df = df.copy()
    
    # Gérer la colonne datetime
    timestamp_col = 'timestamp' if 'timestamp' in processed_df.columns else 'datetime_utc'
    
    # S'assurer que le timestamp est au format datetime
    if processed_df[timestamp_col].dtype != 'datetime64[ns]':
        processed_df[timestamp_col] = pd.to_datetime(processed_df[timestamp_col])
    
    # Pour la compatibilité, toujours avoir une colonne 'timestamp'
    if timestamp_col != 'timestamp':
        processed_df['timestamp'] = processed_df[timestamp_col]
    
    # Ajouter des caractéristiques temporelles si elles ne sont pas déjà présentes
    if 'hour' not in processed_df.columns:
        processed_df['hour'] = processed_df[timestamp_col].dt.hour
    if 'day_of_week' not in processed_df.columns and 'dayofweek' not in processed_df.columns:
        processed_df['day_of_week'] = processed_df[timestamp_col].dt.dayofweek
    
    # Mapper les noms des colonnes pour la compatibilité
    # Cela permet de mapper les colonnes personnalisées aux noms de colonnes standards attendus par le modèle
    col_mapping = {
        'WeatherStation.Weather.Ta': 'outside_temp',
        'cool_elec': 'hvac_power',
        'PV': 'lighting_power',
        'CHP': 'plug_load_power'
    }
    
    # Ajouter les colonnes mappées si elles n'existent pas déjà
    for custom_col, standard_col in col_mapping.items():
        if custom_col in processed_df.columns and standard_col not in processed_df.columns:
            processed_df[standard_col] = processed_df[custom_col]
    
    # Si nous n'avons pas de colonne indoor_temp, l'estimer à partir de la température extérieure
    if 'indoor_temp' not in processed_df.columns and 'outside_temp' in processed_df.columns:
        processed_df['indoor_temp'] = processed_df['outside_temp'] + 19.0  # Supposant une différence d'environ 19°C
    
    # Si nous n'avons pas d'humidité intérieure, utiliser une valeur par défaut
    if 'indoor_humidity' not in processed_df.columns:
        processed_df['indoor_humidity'] = 40.0
    
    # Si nous n'avons pas de niveau de lumière, l'estimer à partir de WeatherStation.Weather.Igm
    if 'light_level' not in processed_df.columns and 'WeatherStation.Weather.Igm' in processed_df.columns:
        # Convertir la radiation solaire en lux approximatifs (estimation très grossière)
        processed_df['light_level'] = processed_df['WeatherStation.Weather.Igm'] * 2.5 + 100
    
    # Si nous n'avons pas de données sur l'occupation, l'estimer en fonction de l'heure (modèle simplifié)
    if 'occupancy' not in processed_df.columns:
        # Les heures de travail sont généralement de 8h à 18h
        processed_df['occupancy'] = 0.0
        work_hours = (processed_df['hour'] >= 8) & (processed_df['hour'] <= 18)
        processed_df.loc[work_hours, 'occupancy'] = 1.0
        
        # Heures de transition (augmentation/diminution)
        morning_transition = (processed_df['hour'] >= 6) & (processed_df['hour'] < 8)
        evening_transition = (processed_df['hour'] > 18) & (processed_df['hour'] <= 20)
        processed_df.loc[morning_transition, 'occupancy'] = (processed_df.loc[morning_transition, 'hour'] - 6) / 2
        processed_df.loc[evening_transition, 'occupancy'] = 1 - (processed_df.loc[evening_transition, 'hour'] - 18) / 2
        
        # Ajustement pour le week-end
        if 'day_of_week' in processed_df.columns:
            weekend = (processed_df['day_of_week'] >= 5)  # 5=Samedi, 6=Dimanche
            processed_df.loc[weekend, 'occupancy'] = processed_df.loc[weekend, 'occupancy'] * 0.3
        elif 'dayofweek' in processed_df.columns:
            weekend = (processed_df['dayofweek'] >= 5)
            processed_df.loc[weekend, 'occupancy'] = processed_df.loc[weekend, 'occupancy'] * 0.3
    
    # Calculer la puissance totale si elle n'est pas présente
    if 'total_power' not in processed_df.columns:
        if 'total' in processed_df.columns:
            processed_df['total_power'] = processed_df['total']
        else:
            # Utiliser les composants énergétiques disponibles
            energy_cols = [col for col in ['hvac_power', 'lighting_power', 'plug_load_power'] 
                          if col in processed_df.columns]
            if energy_cols:
                processed_df['total_power'] = processed_df[energy_cols].sum(axis=1)
    
    # Remplir les valeurs manquantes (si présentes)
    processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
    
    # S'assurer que toutes les colonnes numériques sont de type float
    numeric_cols = processed_df.select_dtypes(include=np.number).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].astype(float)
    
    return processed_df

def calculate_metrics(original_df, optimized_df):
    """
    Calculer les métriques d'optimisation de l'énergie.
    
    Arguments :
        original_df (DataFrame) : Données originales
        optimized_df (DataFrame) : Données optimisées
        
    Retourne :
        dict : Métriques comprenant les économies d'énergie, le score de confort et les économies de coût
    """
    # Calculer la consommation d'énergie
    original_energy = original_df['total_power'].sum()
    optimized_energy = optimized_df['adjusted_total_power'].sum()
    
    # Calculer l'énergie économisée
    energy_saved = original_energy - optimized_energy
    energy_saved_percent = (energy_saved / original_energy) * 100 if original_energy > 0 else 0
    
    # Calculer le score de confort
    comfort_score = optimized_df['comfort_score'].mean()
    
    # Calculer les économies de coût (en supposant $0.15 par kWh)
    cost_per_kwh = 0.15
    hours = len(original_df)
    
    # Convertir la puissance (kW) en énergie (kWh) en multipliant par les heures
    # Supposant que chaque ligne représente 1 heure de données
    original_energy_kwh = original_energy / hours
    optimized_energy_kwh = optimized_energy / hours
    
    cost_saved = (original_energy_kwh - optimized_energy_kwh) * cost_per_kwh * hours
    
    return {
        'energy_saved': energy_saved_percent,
        'comfort_score': comfort_score,
        'cost_saved': cost_saved
    }
