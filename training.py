import numpy as np
import torch
import time
from tqdm import tqdm

def train_ppo(env, agent, epochs=1, episodes_per_epoch=1, max_timesteps=100000):
    """
    Entraîner un agent PPO dans l'environnement donné
    
    Arguments :
        env : L'environnement dans lequel entraîner
        agent : L'agent PPO à entraîner
        epochs : Nombre d'époques d'entraînement
        episodes_per_epoch : Nombre d'épisodes par époque
        max_timesteps : Nombre maximum de pas de temps par épisode
    
    Retourne :
        dict : Metrics d'entraînement
    """
    # Metrics
    total_rewards = []
    energy_saved_percents = []
    comfort_scores = []
    
    # Entraîner pour le nombre d'époques spécifié
    for epoch in range(epochs):
        epoch_rewards = []
        epoch_energy_saved = []
        epoch_comfort = []
        
        # Entraîner pour plusieurs épisodes par époque
        for episode in range(episodes_per_epoch):
            # Réinitialiser l'environnement
            state = env.reset()
            episode_reward = 0
            episode_energy_saved = []
            episode_comfort = []
            done = False
            timestep = 0
            
            # Exécuter l'épisode
            while not done and timestep < max_timesteps:
                # Sélectionner l'action
                action = agent.select_action(state)
                
                # Faire un pas dans l'environnement
                next_state, reward, done, info = env.step(action)
                
                # Stocker la transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Mettre à jour l'état et les compteurs
                state = next_state
                episode_reward += reward
                timestep += 1
                
                # Stocker les métriques
                episode_energy_saved.append(info["energy_saved_percent"])
                episode_comfort.append(info["comfort_score"])
            
            # Mettre à jour la politique après l'épisode
            agent.update()
            
            # Enregistrer les métriques de l'épisode
            epoch_rewards.append(episode_reward)
            epoch_energy_saved.append(np.mean(episode_energy_saved))
            epoch_comfort.append(np.mean(episode_comfort))
        
        # Enregistrer les métriques de l'époque
        total_rewards.append(np.mean(epoch_rewards))
        energy_saved_percents.append(np.mean(epoch_energy_saved))
        comfort_scores.append(np.mean(epoch_comfort))
    
    # Retourner les métriques
    return {
        "avg_reward": np.mean(total_rewards),
        "energy_saved": np.mean(energy_saved_percents),
        "comfort_score": np.mean(comfort_scores)
    }
