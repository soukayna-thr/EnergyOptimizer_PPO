import numpy as np
import gym
from gym import spaces

class SmartBuildingEnv(gym.Env):
    """
    Un environnement d'apprentissage par renforcement pour l'optimisation énergétique d'un bâtiment intelligent.

    Cet environnement simule un bâtiment intelligent avec contrôle du chauffage/climatisation (HVAC), de l’éclairage et des appareils,
    permettant à un agent d’apprendre des stratégies de contrôle optimales qui équilibrent la consommation d’énergie
    et le confort des occupants.
    """
    
    def __init__(self, config):
        super(SmartBuildingEnv, self).__init__()
        
        # Enregistrer la configuration
        self.data = config["data"]
        self.comfort_weight = config["comfort_weight"]
        self.temp_range = config["temp_range"]
        self.light_range = config["light_range"]
        
        # Index actuel des données
        self.current_idx = 0
        self.max_idx = len(self.data) - 1
        
        # Définir l'espace d'action : [ajustement_hvac, ajustement_lumière]
        # HVAC : de -1.0 à 1.0 (réduction/augmentation jusqu'à 20 %)
        # Lumière : de -1.0 à 1.0 (réduction/augmentation jusqu'à 20 %)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Définir l’espace d’observation
        # [temp_ext, temp_int, humidité_int, niveau_lumière, occupation,
        #  heure_sin, heure_cos, jour_semaine_sin, jour_semaine_cos]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        # Définir l'état
        self.state = None
        
        # Coût de l’énergie en dollars par kWh
        self.energy_cost = 0.15
        
        # Réinitialiser l’environnement
        self.reset()
    
    def reset(self):
        """Réinitialise l’environnement au début des données."""
        self.current_idx = 0
        self.state = self._get_observation(self.current_idx)
        return self.state
    
    def step(self, action):
        """
        Effectue une action dans l’environnement avec l’action donnée.

        Args :
            action (array) : [ajustement_hvac, ajustement_lumière]

        Retourne :
            tuple : (état_suivant, récompense, terminé, infos)
        """
        # Obtenir les données actuelles
        current_data = self.data.iloc[self.current_idx]
        
        # Appliquer les actions (ajuster la puissance du HVAC et de la lumière)
        hvac_adjustment = action[0] * 0.2  # Échelle de 20%
        lighting_adjustment = action[1] * 0.2  # Échelle de 20%
        
        # Calculer la consommation ajustée de puissance
        adjusted_hvac_power = current_data["hvac_power"] * (1 + hvac_adjustment)
        adjusted_lighting_power = current_data["lighting_power"] * (1 + lighting_adjustment)
        
        # Calculer la température intérieure ajustée en fonction du changement HVAC
        # Modèle simplifié : réduire le chauffage en hiver diminue la température, en été l'augmente
        is_heating = current_data["outside_temp"] < current_data["indoor_temp"]
        temp_change_direction = -1 if is_heating else 1
        temp_change_magnitude = abs(hvac_adjustment) * 0.5  # Changement max de 0.5°C par étape
        
        adjusted_indoor_temp = current_data["indoor_temp"]
        if hvac_adjustment < 0:  # Réduction de la puissance du HVAC
            adjusted_indoor_temp += temp_change_direction * temp_change_magnitude
        
        # Calculer le niveau de lumière ajusté en fonction du changement d’éclairage
        light_change = lighting_adjustment * current_data["light_level"]
        adjusted_light_level = max(0, current_data["light_level"] + light_change)
        
        # Calculer la puissance totale ajustée
        original_total_power = current_data["total_power"]
        adjusted_total_power = (
            adjusted_hvac_power + 
            adjusted_lighting_power + 
            current_data["plug_load_power"]
        )
        
        # Calculer les économies d’énergie
        energy_saved = original_total_power - adjusted_total_power
        
        # Calculer le score de confort
        temp_comfort = self._calculate_temp_comfort(adjusted_indoor_temp)
        light_comfort = self._calculate_light_comfort(adjusted_light_level, current_data["occupancy"])
        comfort_score = (temp_comfort + light_comfort) / 2
        
        # Calculer la récompense
        energy_reward = energy_saved / original_total_power if original_total_power > 0 else 0
        reward = (1 - self.comfort_weight) * energy_reward + self.comfort_weight * comfort_score
        
        # Passer à l'état suivant
        self.current_idx += 1
        done = self.current_idx >= self.max_idx
        
        if not done:
            self.state = self._get_observation(self.current_idx)
        
        # Créer un dictionnaire d'informations
        info = {
            "timestamp": current_data["timestamp"],
            "outside_temp": current_data["outside_temp"],
            "original_hvac_power": current_data["hvac_power"],
            "adjusted_hvac_power": adjusted_hvac_power,
            "original_lighting_power": current_data["lighting_power"],
            "adjusted_lighting_power": adjusted_lighting_power,
            "plug_load_power": current_data["plug_load_power"],
            "original_indoor_temp": current_data["indoor_temp"],
            "adjusted_indoor_temp": adjusted_indoor_temp,
            "original_light_level": current_data["light_level"],
            "adjusted_light_level": adjusted_light_level,
            "occupancy": current_data["occupancy"],
            "original_total_power": original_total_power,
            "adjusted_total_power": adjusted_total_power,
            "energy_saved": energy_saved,
            "energy_saved_percent": (energy_saved / original_total_power) * 100 if original_total_power > 0 else 0,
            "comfort_score": comfort_score * 100,
            "hvac_adjustment": hvac_adjustment,
            "lighting_adjustment": lighting_adjustment,
            "reward": reward
        }
        
        return self.state, reward, done, info
    
    def _get_observation(self, idx):
        """Convertir la ligne de données actuelle en un vecteur d’observation."""
        data_row = self.data.iloc[idx]
        
        # Extraire l'horodatage pour obtenir l'heure et le jour de la semaine
        timestamp = data_row["timestamp"]
        if isinstance(timestamp, str):
            import pandas as pd
            timestamp = pd.to_datetime(timestamp)
            
        # Encoder les caractéristiques temporelles en utilisant le sinus et le cosinus pour maintenir la nature cyclique
        hour = timestamp.hour if hasattr(timestamp, 'hour') else 0
        day_of_week = timestamp.dayofweek if hasattr(timestamp, 'dayofweek') else 0
        
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Créer le vecteur d’observation
        observation = np.array([
            data_row["outside_temp"],
            data_row["indoor_temp"],
            data_row["indoor_humidity"],
            data_row["light_level"],
            data_row["occupancy"],
            hour_sin,
            hour_cos,
            day_sin,
            day_cos
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_temp_comfort(self, temperature):
        """Calculer le score de confort thermique (0-1)."""
        min_temp, max_temp = self.temp_range
        
        # Confort parfait si dans la plage
        if min_temp <= temperature <= max_temp:
            return 1.0
        
        # Confort réduit en dehors de la plage, avec une chute linéaire
        if temperature < min_temp:
            # Le confort diminue à mesure que la température descend en dessous du minimum
            # À 3 degrés en dessous du minimum, le confort est 0
            return max(0, 1 - (min_temp - temperature) / 3)
        else:
            # Le confort diminue à mesure que la température dépasse le maximum
            # À 3 degrés au-dessus du maximum, le confort est 0
            return max(0, 1 - (temperature - max_temp) / 3)
    
    def _calculate_light_comfort(self, light_level, occupancy):
        """Calculer le score de confort lumineux (0-1)."""
        min_light, max_light = self.light_range
        
        # Si aucune occupation, l’éclairage importe peu pour le confort
        if occupancy < 0.1:
            return 1.0
        
        # Confort parfait si dans la plage
        if min_light <= light_level <= max_light:
            return 1.0
        
        # Confort réduit en dehors de la plage, avec une chute linéaire
        if light_level < min_light:
            # À 200 lux en dessous du minimum, le confort est 0
            return max(0, 1 - (min_light - light_level) / 200)
        else:
            # À 300 lux au-dessus du maximum, le confort est 0
            return max(0, 1 - (light_level - max_light) / 300)
    
    def get_state(self):
        """Retourner l’état actuel."""
        return self.state
