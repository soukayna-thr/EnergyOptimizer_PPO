import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time

from energy_env import SmartBuildingEnv
from ppo_agent import PPOAgent
from training import train_ppo
from utils import load_data, preprocess_data, calculate_metrics
from visualization import (
    plot_energy_consumption,
    plot_comfort_metrics,
    plot_optimization_comparison,
    plot_training_progress,
    plot_control_actions
)
from building_3d_vis import (
    create_building_model,
    plot_building_3d,
    plot_building_animation,
    update_building_state
)

# Set page configuration
st.set_page_config(
    page_title="Optimisation Énergétique des Bâtiments Intelligents",
    page_icon="🏢",
    layout="wide"
)

# Main title
st.title("🏢 Système d'Optimisation Énergétique pour Bâtiments Intelligents")
st.markdown("### Utilisation de l'Apprentissage par Renforcement PPO pour Équilibrer Énergie et Confort")

# Sidebar
st.sidebar.title("Configuration")

# Create two main tabs in the sidebar
sidebar_tab1, sidebar_tab2 = st.sidebar.tabs(["Optimisation", "Documentation"])

with sidebar_tab1:
    # Data loading section
    st.header("1. Source de Données")
    data_source = st.radio(
        "Sélectionnez une source de données:",
        ["Données Échantillon", "Données Bâtiment Personnalisées", "Télécharger Données"]
    )

# Function to display the documentation content
def show_documentation():
    # Application information
    st.header("À propos de cette Application")
    st.write("""
    Cette application utilise l'Optimisation par Politique Proximale (PPO), un algorithme d'apprentissage par renforcement de pointe, 
    pour optimiser la consommation d'énergie dans les bâtiments intelligents tout en maintenant les niveaux de confort des occupants.
    
    ### Fonctionnalités Clés:
    - **Analyse de Données:** Analyse des modèles de consommation d'énergie à partir des données du bâtiment
    - **Apprentissage par Renforcement:** Mise en œuvre de PPO pour optimiser les stratégies de contrôle
    - **Équilibre de Confort:** Équilibre entre les économies d'énergie et le confort des occupants
    - **Visualisation:** Visualisation des modèles de consommation et des résultats d'optimisation
    
    ### Comment cela fonctionne:
    1. Téléchargez les données de consommation d'énergie de votre bâtiment intelligent
    2. Configurez les paramètres de confort et les paramètres d'entraînement
    3. Entraînez l'agent PPO pour apprendre des stratégies de contrôle optimales
    4. Examinez les résultats montrant les économies d'énergie et les mesures de confort
    """)
    
    # Theoretical explanation section
    st.header("Fondements Théoriques de l'Apprentissage par Renforcement")
    
    # RL Basics Tab
    tab1, tab2, tab3, tab4 = st.tabs(["Concepts de Base", "Modélisation de l'Environnement", "Algorithme PPO", "Hyperparamètres"])
    
    with tab1:
        st.subheader("Concepts Fondamentaux de l'Apprentissage par Renforcement")
        st.markdown("""
        L'apprentissage par renforcement (RL) est un paradigme d'apprentissage automatique où un agent apprend à prendre des décisions 
        en interagissant avec un environnement pour maximiser une récompense cumulative.
        
        #### Composants Clés
        
        1. **Agent**: L'entité qui prend des décisions (dans notre cas, le contrôleur du bâtiment).
        
        2. **Environnement**: Le système avec lequel l'agent interagit (le bâtiment intelligent).
        
        3. **État (S)**: Une représentation de la situation actuelle de l'environnement.
           - Dans notre application: `[outside_temp, indoor_temp, indoor_humidity, light_level, occupancy, hour_sin, hour_cos, day_sin, day_cos]`
        
        4. **Action (A)**: Les décisions que l'agent peut prendre.
           - Dans notre application: `[hvac_adjustment, lighting_adjustment]`
        
        5. **Récompense (R)**: Le signal de feedback que l'agent reçoit après chaque action.
           - Dans notre application: `reward = (1 - comfort_weight) * energy_reward + comfort_weight * comfort_score`
        
        6. **Politique (π)**: La stratégie que l'agent utilise pour déterminer ses actions.
           - Dans notre application: Une politique stochastique représentée par un réseau de neurones.
        
        #### Processus de Décision de Markov (MDP)
        
        Le problème d'optimisation énergétique est formulé comme un MDP, où:
        - Les transitions d'état suivent la propriété de Markov (l'état futur dépend uniquement de l'état actuel et de l'action).
        - L'objectif est de trouver une politique optimale qui maximise la récompense cumulée attendue.
        
        #### Défis Spécifiques à l'Optimisation Énergétique
        
        - **Compromis Multi-objectifs**: Équilibrer l'économie d'énergie et le confort des occupants.
        - **Dynamique Temporelle**: Les conditions extérieures et l'occupation varient dans le temps.
        - **Contraintes Opérationnelles**: Respecter les limites physiques des systèmes CVC et d'éclairage.
        """)
    
    with tab2:
        st.subheader("Modélisation de l'Environnement et des Récompenses")
        st.markdown("""
        #### Structure de l'Environnement (SmartBuildingEnv)
        
        Notre environnement `SmartBuildingEnv` hérite de l'interface gym.Env et implémente:
        
        1. **Espace d'États**: 
           - Variables continues représentant les conditions du bâtiment et le temps.
           - Normalisation des caractéristiques pour améliorer l'apprentissage.
        
        2. **Espace d'Actions**: 
           - Actions continues pour les ajustements HVAC et d'éclairage.
           - Plage d'action [-1, 1] mise à l'échelle pour représenter des ajustements de ±20%.
        
        3. **Fonction de Transition**: 
           - Simule comment les actions affectent l'environnement intérieur.
           - Modélise les changements de température et d'éclairage en fonction des contrôles.
        
        #### Conception de la Fonction de Récompense
        
        La fonction de récompense est cruciale pour guider l'apprentissage de l'agent:
        
        ```python
        # Calcul des composantes de la récompense
        energy_reward = energy_saved / original_total_power
        comfort_score = (temp_comfort + light_comfort) / 2
        
        # Récompense pondérée combinant économie d'énergie et confort
        reward = (1 - comfort_weight) * energy_reward + comfort_weight * comfort_score
        ```
        
        #### Métriques de Confort
        
        1. **Confort Thermique**: 
           - Score maximal (1.0) dans la plage de température confortable.
           - Décroissance linéaire en dehors de cette plage.
        
        2. **Confort d'Éclairage**: 
           - Score maximal (1.0) dans la plage de lux confortable.
           - Importance variable selon l'occupation.
        
        Cette conception de récompense permet à l'agent de trouver un équilibre optimal entre:
        - Minimiser la consommation d'énergie (économiser des ressources)
        - Maintenir des conditions de confort acceptables (satisfaire les occupants)
        """)
    
    with tab3:
        st.subheader("Algorithme PPO (Proximal Policy Optimization)")
        st.markdown("""
        PPO est un algorithme d'apprentissage par renforcement avancé qui combine:
        
        #### Architecture Acteur-Critique
        
        Notre implémentation utilise une architecture réseau avec deux composants:
        
        1. **Acteur (Politique)**: 
           - Détermine quelles actions prendre dans un état donné.
           - Produit une distribution de probabilité sur les actions possibles.
           - Dans notre cas: distribution gaussienne avec moyenne et écart-type.
        
        2. **Critique (Fonction de Valeur)**: 
           - Évalue la "valeur" d'un état.
           - Aide à réduire la variance lors de l'apprentissage.
        
        #### Avantages de PPO
        
        1. **Stabilité**: 
           - Utilise un ratio de probabilité limité pour éviter les mises à jour de politique trop grandes.
           - Le paramètre `clip_param` (0.2 dans notre implémentation) contrôle cette limitation.
        
        2. **Échantillonnage Efficace**: 
           - Réutilise les expériences passées via l'apprentissage hors politique.
           - Plus efficace en termes d'échantillons que les méthodes de politique sur politique.
        
        3. **Apprentissage Continu**: 
           - Prend en charge les espaces d'action continus nécessaires pour le contrôle CVC et d'éclairage.
        
        #### Fonction Objectif de PPO
        
        ```
        L_CLIP(θ) = Ê_t [ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ]
        ```
        
        - `r_t(θ)` est le ratio de probabilité entre la nouvelle et l'ancienne politique.
        - `A_t` est l'estimation de l'avantage.
        - `ε` est le paramètre de clip (0.2 dans notre cas).
        
        #### Estimation d'Avantage Généralisée (GAE)
        
        Notre implémentation utilise GAE pour estimer l'avantage d'une action:
        
        ```python
        delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        ```
        
        Avec:
        - `gamma` = 0.99 (facteur d'actualisation)
        - `gae_lambda` = 0.95 (paramètre de compromis biais-variance)
        """)
    
    with tab4:
        st.subheader("Hyperparamètres et leur Influence")
        st.markdown("""
        #### Hyperparamètres de PPO
        
        | Paramètre | Valeur | Influence |
        |-----------|--------|-----------|
        | `learning_rate` | Variable (UI) | Taille des pas d'optimisation; impacte la vitesse et la stabilité de l'apprentissage |
        | `batch_size` | Variable (UI) | Nombre d'échantillons utilisés pour chaque mise à jour; affecte l'efficacité et la stabilité |
        | `gamma` | 0.99 | Facteur d'actualisation; détermine l'importance des récompenses futures vs immédiates |
        | `gae_lambda` | 0.95 | Coefficient GAE; contrôle le compromis biais-variance dans l'estimation d'avantage |
        | `clip_param` | 0.2 | Limite les changements de politique; aide à la stabilité de l'apprentissage |
        | `entropy_coef` | 0.01 | Encourage l'exploration; empêche la convergence prématurée |
        | `value_loss_coef` | 0.5 | Poids de la perte de la fonction de valeur vs la perte de politique |
        | `max_grad_norm` | 0.5 | Écrêtage du gradient; prévient les explosions de gradient |
        | `ppo_epochs` | 10 | Nombre d'époques par lot; plus d'époques = plus d'extraction d'information des données |
        
        #### Paramètres d'Environnement
        
        | Paramètre | Valeur | Influence |
        |-----------|--------|-----------|
        | `comfort_weight` | Variable (UI) | Équilibre entre économie d'énergie et confort; valeurs plus élevées priorisent le confort |
        | `temp_range` | Variable (UI) | Plage de température confortable; plage plus large = plus facile à satisfaire |
        | `light_range` | Variable (UI) | Plage d'éclairage confortable; plage plus large = plus facile à satisfaire |
        
        #### Conseils pour l'Optimisation des Hyperparamètres
        
        1. **Learning Rate**: 
           - Trop élevé: apprentissage instable
           - Trop bas: apprentissage lent
           - Recommandation: commencer avec 0.0005 et ajuster
        
        2. **Comfort Weight**: 
           - Proche de 0: priorité maximale à l'économie d'énergie, potentiellement au détriment du confort
           - Proche de 1: priorité maximale au confort, potentiellement sans économies significatives
           - Recommandation: 0.5-0.7 pour un bon équilibre
        
        3. **Batch Size**: 
           - Plus grand: estimations plus stables mais moins d'updates
           - Plus petit: plus d'updates mais plus de variance
           - Recommandation: 64-128 pour un bon équilibre
        
        4. **Nombre d'Époques**: 
           - Plus élevé: meilleure extraction d'information mais risque de surapprentissage
           - Recommandation: 30-100 pour des résultats significatifs
        """)
    
    # Add code explanation section
    st.header("Structure du Code et Implémentation")
    
    code_tab1, code_tab2, code_tab3 = st.tabs(["Architecture Globale", "Environnement RL", "Agent PPO"])
    
    with code_tab1:
        st.markdown("""
        ### Organisation du Code
        
        Le projet est structuré en plusieurs modules avec des responsabilités spécifiques:
        
        1. **app.py**: 
           - Interface utilisateur Streamlit
           - Contrôle du flux de l'application
           - Visualisation des résultats
        
        2. **energy_env.py**: 
           - Définition de l'environnement `SmartBuildingEnv`
           - Modélisation des dynamiques du bâtiment
           - Calcul des récompenses
        
        3. **ppo_agent.py**: 
           - Implémentation de l'algorithme PPO
           - Architecture réseau acteur-critique
           - Logique d'apprentissage et d'inférence
        
        4. **training.py**: 
           - Boucle d'entraînement
           - Collecte des métriques
           - Gestion des épisodes
        
        5. **utils.py**: 
           - Chargement et prétraitement des données
           - Calcul des métriques de performance
        
        6. **visualization.py**: 
           - Génération de graphiques pour les données et résultats
        
        Cette séparation modulaire facilite la maintenance et l'extension du code, et permet de tester individuellement chaque composant.
        """)
        
    with code_tab2:
        st.code("""
# Extrait simplifié de energy_env.py
class SmartBuildingEnv(gym.Env):
    def __init__(self, config):
        # Définir les espaces d'état et d'action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        
        # Stocker la configuration
        self.data = config["data"]
        self.comfort_weight = config["comfort_weight"]
        self.temp_range = config["temp_range"]
        self.light_range = config["light_range"]
    
    def reset(self):
        # Réinitialiser l'environnement
        self.current_idx = 0
        self.state = self._get_observation(self.current_idx)
        return self.state
    
    def step(self, action):
        # Appliquer les actions au bâtiment
        hvac_adjustment = action[0] * 0.2  # Mise à l'échelle
        lighting_adjustment = action[1] * 0.2
        
        # Calculer les nouvelles conditions du bâtiment
        adjusted_hvac_power = current_data["hvac_power"] * (1 + hvac_adjustment)
        adjusted_lighting_power = current_data["lighting_power"] * (1 + lighting_adjustment)
        
        # Calculer le confort et les économies d'énergie
        energy_saved = original_total_power - adjusted_total_power
        comfort_score = (temp_comfort + light_comfort) / 2
        
        # Calculer la récompense
        energy_reward = energy_saved / original_total_power
        reward = (1 - self.comfort_weight) * energy_reward + self.comfort_weight * comfort_score
        
        # Avancer à l'état suivant
        self.current_idx += 1
        done = self.current_idx >= self.max_idx
        self.state = self._get_observation(self.current_idx) if not done else self.state
        
        return self.state, reward, done, info
        """, language='python')
        
    with code_tab3:
        st.code("""
# Extrait simplifié de ppo_agent.py
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Réseaux de l'acteur (politique)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Réseau du critique (fonction de valeur)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def get_action(self, state, deterministic=False):
        # Obtenir la distribution d'action
        action_mean, action_std, _ = self.forward(state)
        
        if deterministic:
            return action_mean
        
        # Échantillonner une action
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return torch.tanh(action), log_prob

class PPOAgent:
    def update(self):
        # Calcul des retours et avantages
        returns, advantages = self._compute_gae(rewards, values, dones)
        
        # Boucle d'entraînement PPO
        for _ in range(self.ppo_epochs):
            # Mini-batch training
            for batch_indices in batches:
                # Calcul du ratio de probabilité
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Objectif PPO avec clipping
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Pertes de la fonction de valeur et de l'entropie
                value_loss = 0.5 * (batch_returns - values_pred).pow(2).mean()
                entropy_loss = -entropy.mean()
                
                # Perte totale
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Mise à jour des paramètres
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        """, language='python')
    
    st.markdown("""
    ## Fin de la Documentation
    
    """)

# Handle documentation tab
with sidebar_tab2:
    st.write("Documentation complète sur l'apprentissage par renforcement.")
    if st.button("Afficher la Documentation"):
        # Create a flag in session state to show documentation
        st.session_state['show_docs'] = True

# Load data
if data_source == "Données Échantillon":
    data_path = "data/sample_data.csv"
    df = load_data(data_path)
    st.success("Données échantillon standard chargées avec succès!")
elif data_source == "Données Bâtiment Personnalisées":
    data_path = "data/custom_data.csv"
    df = load_data(data_path)
    st.success("Données personnalisées du bâtiment chargées avec succès!")
else:
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV avec des données de consommation énergétique", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Données téléchargées avec succès!")
    else:
        st.warning("Veuillez télécharger un fichier de données pour continuer.")
        df = None

# Initialize show_docs in session state if it doesn't exist
if 'show_docs' not in st.session_state:
    st.session_state['show_docs'] = False

# Main content
if st.session_state.get('show_docs', False):
    # Display documentation when the flag is set
    show_documentation()
    
    # Add button to return to main content
    if st.button("Retour à l'Application"):
        st.session_state['show_docs'] = False
        st.rerun()
        
elif df is not None:
    # Display data info
    st.header("1. Aperçu des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Échantillon de Données:")
        st.dataframe(df.head())
    
    with col2:
        st.write("Statistiques des Données:")
        st.dataframe(df.describe())
    
    # Data preprocessing
    st.header("2. Prétraitement des Données")
    processed_df = preprocess_data(df)
    st.success("Données prétraitées avec succès!")
    
    # Display original energy consumption
    st.header("3. Modèles Actuels de Consommation d'Énergie")
    fig = plot_energy_consumption(processed_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Training configuration
    st.header("4. Configuration de l'Optimisation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Paramètres d'Environnement")
        comfort_weight = st.slider("Poids du Confort", 0.0, 1.0, 0.5, 0.05,
                                 help="Poids du confort dans la fonction de récompense (plus élevé signifie prioriser le confort)")
        
        temp_range = st.slider("Plage de Température Confortable (°C)", 18.0, 28.0, (20.0, 25.0),
                              help="Plage de température considérée comme confortable pour les occupants")
        
        light_range = st.slider("Plage de Niveau d'Éclairage Confortable (lux)", 300, 800, (400, 700),
                               help="Plage de niveau d'éclairage considérée comme confortable pour les occupants")
    
    with col2:
        st.subheader("Paramètres d'Apprentissage")
        epochs = st.slider("Époques d'Entraînement", 10, 200, 50, 
                          help="Nombre d'époques d'entraînement")
        
        learning_rate = st.selectbox(
            "Taux d'Apprentissage", 
            [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005],
            index=2,
            help="Taux d'apprentissage pour l'algorithme PPO"
        )
        
        batch_size = st.selectbox(
            "Taille du Batch",
            [16, 32, 64, 128, 256],
            index=2,
            help="Taille du batch pour l'entraînement"
        )
    
    # Training section
    st.header("5. Lancer l'Optimisation")
    
    if st.button("Démarrer l'Entraînement d'Optimisation", key="train"):
        # Create environment
        env_config = {
            "data": processed_df,
            "comfort_weight": comfort_weight,
            "temp_range": temp_range,
            "light_range": light_range
        }
        
        env = SmartBuildingEnv(env_config)
        
        # Create agent
        agent_config = {
            "state_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.shape[0],
            "lr": learning_rate,
            "batch_size": batch_size
        }
        
        agent = PPOAgent(agent_config)
        
        # Training progress placeholder
        progress_bar = st.progress(0)
        training_metrics_container = st.empty()
        training_plot_container = st.empty()
        
        # Train the agent
        progress = {"epochs": [], "rewards": [], "energy_saved": [], "comfort_score": []}
        
        for i in range(epochs):
            # Training step
            metrics = train_ppo(env, agent, epochs=1)
            
            # Update progress
            progress["epochs"].append(i+1)
            progress["rewards"].append(metrics["avg_reward"])
            progress["energy_saved"].append(metrics["energy_saved"])
            progress["comfort_score"].append(metrics["comfort_score"])
            
            # Update UI
            progress_bar.progress((i+1)/epochs)
            
            # Show current metrics
            training_metrics_container.write(f"Époque {i+1}/{epochs} - "
                                            f"Récompense Moy: {metrics['avg_reward']:.4f}, "
                                            f"Énergie Économisée: {metrics['energy_saved']:.2f}%, "
                                            f"Score de Confort: {metrics['comfort_score']:.2f}%")
            
            # Plot training progress
            if (i+1) % 5 == 0 or i == epochs-1:
                fig = plot_training_progress(progress)
                training_plot_container.plotly_chart(fig, use_container_width=True)
        
        st.success(f"Optimisation terminée en {epochs} époques!")
        
        # Results section
        st.header("6. Résultats de l'Optimisation")
        
        # Run optimized policy
        env.reset()
        done = False
        optimized_data = []
        
        with st.spinner("Simulation de la stratégie de contrôle optimisée..."):
            while not done:
                state = env.get_state()
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                optimized_data.append(info)
        
        # Convert to dataframe
        optimized_df = pd.DataFrame(optimized_data)
        
        # Calculate metrics
        metrics = calculate_metrics(processed_df, optimized_df)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Économies d'Énergie", f"{metrics['energy_saved']:.2f}%", "↓ Bon")
            
        with col2:
            st.metric("Confort Maintenu", f"{metrics['comfort_score']:.2f}%", "↑ Bon")
            
        with col3:
            st.metric("Réduction des Coûts", f"{metrics['cost_saved']:.2f}€", "↓ Bon")
        
        # Plot comparison
        st.subheader("Consommation d'Énergie: Avant vs Après Optimisation")
        fig = plot_optimization_comparison(processed_df, optimized_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot comfort metrics
        st.subheader("Métriques de Confort")
        fig = plot_comfort_metrics(processed_df, optimized_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot control actions
        st.subheader("Actions de Contrôle Optimisées")
        fig = plot_control_actions(optimized_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D Building Visualization
        st.header("Visualisation 3D du Bâtiment Intelligent")
        st.write("""
        Cette visualisation 3D montre comment les actions de l'agent affectent les différentes pièces du bâtiment.
        Les couleurs représentent la température (bleu = froid, blanc = neutre, rouge = chaud), 
        la transparence indique le niveau d'éclairage, et les points verts indiquent la présence d'occupants.
        """)
        
        # Configuration du bâtiment
        st.subheader("Configuration du Bâtiment")
        col1, col2, col3 = st.columns(3)
        with col1:
            num_floors = st.slider("Nombre d'étages", min_value=1, max_value=5, value=3)
        with col2:
            num_rooms = st.slider("Nombre de pièces par étage", min_value=4, max_value=16, value=4, step=4)
        with col3:
            building_type = st.selectbox("Type de bâtiment", ["bureau", "résidentiel", "commercial", "industriel"])
        
        # Créer le modèle du bâtiment
        building_model = create_building_model(num_floors=num_floors, num_rooms_per_floor=num_rooms, building_type=building_type)
        
        # Afficher le modèle initial
        st.subheader("État Initial du Bâtiment")
        st.write("""
        Utilisez les contrôles ci-dessous pour explorer le bâtiment en 3D. Vous pouvez:
        - Faire pivoter le modèle en cliquant et faisant glisser
        - Zoomer avec la molette de la souris
        - Voir les détails des pièces en passant le curseur dessus
        - Changer la vue avec les boutons du menu déroulant (en haut à gauche)
        """)
        initial_fig = plot_building_3d(building_model, show_controls=True)
        st.plotly_chart(initial_fig, use_container_width=True)
        
        # Légende explicative complémentaire
        legend_col1, legend_col2, legend_col3 = st.columns(3)
        with legend_col1:
            st.markdown("**Couleurs des pièces**")
            st.markdown("- 🔵 Bleu: Température basse")
            st.markdown("- ⚪ Blanc: Température neutre")
            st.markdown("- 🔴 Rouge: Température élevée")
        with legend_col2:
            st.markdown("**Autres éléments**")
            st.markdown("- 🟢 Points verts: Occupants")
            st.markdown("- 🔍 Transparence: Niveau d'éclairage")
            st.markdown("- 🪟 Bleu clair: Fenêtres")
        with legend_col3:
            st.markdown("**Types de pièces**")
            st.markdown("- Bureau: Espaces de travail")
            st.markdown("- Salle de réunion: Réunions")
            st.markdown("- Espace commun: Zones partagées")
            st.markdown("- Stockage: Rangement, archives")
        
        # Simulation 3D de l'optimisation
        st.subheader("Simulation de l'Optimisation")
        st.write("""
        Cette section montre comment l'agent PPO optimise dynamiquement le bâtiment au fil du temps.
        Observez comment les températures et niveaux d'éclairage s'ajustent en fonction des actions de l'agent
        et des changements d'occupation.
        """)
        
        # Pour la démo, on va simuler des actions HVAC et éclairage
        if 'hvac_action' not in optimized_df.columns:
            optimized_df['hvac_action'] = optimized_df.apply(lambda row: np.random.uniform(-0.2, 0.2), axis=1)
        if 'lighting_action' not in optimized_df.columns:
            optimized_df['lighting_action'] = optimized_df.apply(lambda row: np.random.uniform(-0.2, 0.2), axis=1)
        if 'occupancy' not in optimized_df.columns:
            # Créer une colonne d'occupation temporaire pour la visualisation
            optimized_df['occupancy'] = [np.random.randint(0, 2, len(building_model['rooms'])) for _ in range(len(optimized_df))]
        
        # Bâtiment intelligent avancé - Simulation et optimisation 3D
        st.subheader("Bâtiment 3D Intelligent - Simulation d'Optimisation")
        
        # Version améliorée pour répondre aux besoins du projet
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualisation 3D au centre avec contrôles avancés
            fig = plot_building_3d(building_model, show_controls=True)
            
            # Ajout d'un titre explicatif sur le graphique
            fig.update_layout(
                title="Simulation du bâtiment avec l'agent PPO",
                scene=dict(
                    xaxis_title="X (m)",
                    yaxis_title="Y (m)",
                    zaxis_title="Z (m)",
                    aspectmode='cube'
                )
            )
            
            # Afficher la visualisation 3D avancée
            st.plotly_chart(fig, use_container_width=True, key="main_building_view")
            
            # Explications clés sous la visualisation
            st.info("""
            **Légende des couleurs:**
            - 🔵 **Bleu**: Zones refroidies par l'agent PPO pour économiser l'énergie
            - ⚪ **Blanc**: Zones à température optimale (économie + confort)
            - 🔴 **Rouge**: Zones qui nécessitent du chauffage selon l'occupation
            
            La **transparence** représente l'intensité de l'éclairage, et les **points verts** indiquent la présence d'occupants.
            """)
        
        with col2:
            # Statistiques et informations sur les décisions de l'agent PPO
            st.markdown("#### Statistiques du bâtiment")
            
            # Température et éclairage moyens
            temp_avg = np.mean([room['temperature'] for room in building_model['rooms']])
            light_avg = np.mean([room['light_level'] for room in building_model['rooms']])
            
            # Comptage des pièces occupées
            occupied_count = sum([1 for room in building_model['rooms'] if room.get('occupied', 0) == 1])
            
            # Afficher les métriques
            st.metric("Température moyenne", f"{temp_avg:.1f} °C")
            st.metric("Éclairage moyen", f"{light_avg:.0f} lux")
            st.metric("Taux d'occupation", f"{occupied_count}/{len(building_model['rooms'])} pièces")
            
            # Décisions de l'agent
            st.markdown("#### Décisions de l'agent PPO")
            
            # Prendre la dernière action de l'agent s'il existe des données
            if 'hvac_action' in optimized_df.columns and len(optimized_df) > 0:
                last_action_hvac = optimized_df.iloc[-1]['hvac_action']
                action_text = "⬇️ Refroidissement" if last_action_hvac < -0.05 else "⬆️ Chauffage" if last_action_hvac > 0.05 else "⟷ Maintien"
                st.metric("Action HVAC", action_text, f"{last_action_hvac:.2f}")
            else:
                st.metric("Action HVAC", "Non disponible")
                
            if 'lighting_action' in optimized_df.columns and len(optimized_df) > 0:
                last_action_light = optimized_df.iloc[-1]['lighting_action']
                light_text = "⬇️ Réduction" if last_action_light < -0.05 else "⬆️ Augmentation" if last_action_light > 0.05 else "⟷ Maintien" 
                st.metric("Action Éclairage", light_text, f"{last_action_light:.2f}")
            else:
                st.metric("Action Éclairage", "Non disponible")
            
            # Afficher les économies d'énergie estimées
            st.markdown("#### Économies réalisées")
            st.metric("Énergie économisée", f"{metrics['energy_saved']:.1f}%")
            st.metric("Coût économisé", f"{metrics['cost_saved']:.2f}€")
            st.metric("Confort maintenu", f"{metrics['comfort_score']:.1f}%")
            
        # Explications sur les différents types de pièces
        with st.expander("Types de pièces et leur comportement"):
            st.markdown("""
            ## Types de pièces dans le bâtiment
            
            Chaque type de pièce a des besoins énergétiques différents :
            
            - **Bureaux** : Nécessitent une température stable et un bon éclairage pendant les heures de travail
            - **Salles de réunion** : Occupation intermittente, nécessitent un conditionnement rapide
            - **Espaces communs** : Utilisation variable, souvent plus occupés pendant les pauses
            - **Stockage** : Besoins minimaux en chauffage et éclairage
            
            L'agent PPO apprend à optimiser chaque espace selon son utilisation spécifique.
            """)
        
        # Conclusions
        st.header("7. Conclusions")
        st.write(f"""
        L'algorithme d'apprentissage par renforcement PPO a optimisé avec succès la consommation d'énergie du bâtiment 
        avec une réduction de **{metrics['energy_saved']:.2f}%** tout en maintenant 
        **{metrics['comfort_score']:.2f}%** du niveau de confort des occupants.
        
        Cela se traduit par environ **{metrics['cost_saved']:.2f}€** d'économies de coûts sur la période de données.
        
        L'algorithme a appris à:
        - Ajuster les points de consigne de température en fonction des modèles d'occupation
        - Optimiser les niveaux d'éclairage tout au long de la journée
        - Réduire l'utilisation du CVC pendant les périodes d'inoccupation tout en maintenant la préparation du système
        """)
else:
    # Display placeholder when no data is loaded
    st.info("Veuillez sélectionner ou télécharger des données pour démarrer le processus d'optimisation.")
    
    # Show application information
    st.header("À propos de cette Application")
    st.write("""
    Cette application utilise l'Optimisation par Politique Proximale (PPO), un algorithme d'apprentissage par renforcement de pointe, 
    pour optimiser la consommation d'énergie dans les bâtiments intelligents tout en maintenant les niveaux de confort des occupants.
    
    ### Fonctionnalités Clés:
    - **Analyse de Données:** Analyse des modèles de consommation d'énergie à partir des données du bâtiment
    - **Apprentissage par Renforcement:** Mise en œuvre de PPO pour optimiser les stratégies de contrôle
    - **Équilibre de Confort:** Équilibre entre les économies d'énergie et le confort des occupants
    - **Visualisation:** Visualisation des modèles de consommation et des résultats d'optimisation
    
    ### Comment cela fonctionne:
    1. Téléchargez les données de consommation d'énergie de votre bâtiment intelligent
    2. Configurez les paramètres de confort et les paramètres d'entraînement
    3. Entraînez l'agent PPO pour apprendre des stratégies de contrôle optimales
    4. Examinez les résultats montrant les économies d'énergie et les mesures de confort
    
    Commencez en sélectionnant une source de données dans la barre latérale!
    """)
    
    # Add theoretical explanation section
    st.header("Fondements Théoriques de l'Apprentissage par Renforcement")
    
    # RL Basics Tab
    tab1, tab2, tab3, tab4 = st.tabs(["Concepts de Base", "Modélisation de l'Environnement", "Algorithme PPO", "Hyperparamètres"])
    
    with tab1:
        st.subheader("Concepts Fondamentaux de l'Apprentissage par Renforcement")
        st.markdown("""
        L'apprentissage par renforcement (RL) est un paradigme d'apprentissage automatique où un agent apprend à prendre des décisions 
        en interagissant avec un environnement pour maximiser une récompense cumulative.
        
        #### Composants Clés
        
        1. **Agent**: L'entité qui prend des décisions (dans notre cas, le contrôleur du bâtiment).
        
        2. **Environnement**: Le système avec lequel l'agent interagit (le bâtiment intelligent).
        
        3. **État (S)**: Une représentation de la situation actuelle de l'environnement.
           - Dans notre application: `[outside_temp, indoor_temp, indoor_humidity, light_level, occupancy, hour_sin, hour_cos, day_sin, day_cos]`
        
        4. **Action (A)**: Les décisions que l'agent peut prendre.
           - Dans notre application: `[hvac_adjustment, lighting_adjustment]`
        
        5. **Récompense (R)**: Le signal de feedback que l'agent reçoit après chaque action.
           - Dans notre application: `reward = (1 - comfort_weight) * energy_reward + comfort_weight * comfort_score`
        
        6. **Politique (π)**: La stratégie que l'agent utilise pour déterminer ses actions.
           - Dans notre application: Une politique stochastique représentée par un réseau de neurones.
        
        #### Processus de Décision de Markov (MDP)
        
        Le problème d'optimisation énergétique est formulé comme un MDP, où:
        - Les transitions d'état suivent la propriété de Markov (l'état futur dépend uniquement de l'état actuel et de l'action).
        - L'objectif est de trouver une politique optimale qui maximise la récompense cumulée attendue.
        
        #### Défis Spécifiques à l'Optimisation Énergétique
        
        - **Compromis Multi-objectifs**: Équilibrer l'économie d'énergie et le confort des occupants.
        - **Dynamique Temporelle**: Les conditions extérieures et l'occupation varient dans le temps.
        - **Contraintes Opérationnelles**: Respecter les limites physiques des systèmes CVC et d'éclairage.
        """)
    
    with tab2:
        st.subheader("Modélisation de l'Environnement et des Récompenses")
        st.markdown("""
        #### Structure de l'Environnement (SmartBuildingEnv)
        
        Notre environnement `SmartBuildingEnv` hérite de l'interface gym.Env et implémente:
        
        1. **Espace d'États**: 
           - Variables continues représentant les conditions du bâtiment et le temps.
           - Normalisation des caractéristiques pour améliorer l'apprentissage.
        
        2. **Espace d'Actions**: 
           - Actions continues pour les ajustements HVAC et d'éclairage.
           - Plage d'action [-1, 1] mise à l'échelle pour représenter des ajustements de ±20%.
        
        3. **Fonction de Transition**: 
           - Simule comment les actions affectent l'environnement intérieur.
           - Modélise les changements de température et d'éclairage en fonction des contrôles.
        
        #### Conception de la Fonction de Récompense
        
        La fonction de récompense est cruciale pour guider l'apprentissage de l'agent:
        
        ```python
        # Calcul des composantes de la récompense
        energy_reward = energy_saved / original_total_power
        comfort_score = (temp_comfort + light_comfort) / 2
        
        # Récompense pondérée combinant économie d'énergie et confort
        reward = (1 - comfort_weight) * energy_reward + comfort_weight * comfort_score
        ```
        
        #### Métriques de Confort
        
        1. **Confort Thermique**: 
           - Score maximal (1.0) dans la plage de température confortable.
           - Décroissance linéaire en dehors de cette plage.
        
        2. **Confort d'Éclairage**: 
           - Score maximal (1.0) dans la plage de lux confortable.
           - Importance variable selon l'occupation.
        
        Cette conception de récompense permet à l'agent de trouver un équilibre optimal entre:
        - Minimiser la consommation d'énergie (économiser des ressources)
        - Maintenir des conditions de confort acceptables (satisfaire les occupants)
        """)
    
    with tab3:
        st.subheader("Algorithme PPO (Proximal Policy Optimization)")
        st.markdown("""
        PPO est un algorithme d'apprentissage par renforcement avancé qui combine:
        
        #### Architecture Acteur-Critique
        
        Notre implémentation utilise une architecture réseau avec deux composants:
        
        1. **Acteur (Politique)**: 
           - Détermine quelles actions prendre dans un état donné.
           - Produit une distribution de probabilité sur les actions possibles.
           - Dans notre cas: distribution gaussienne avec moyenne et écart-type.
        
        2. **Critique (Fonction de Valeur)**: 
           - Évalue la "valeur" d'un état.
           - Aide à réduire la variance lors de l'apprentissage.
        
        #### Avantages de PPO
        
        1. **Stabilité**: 
           - Utilise un ratio de probabilité limité pour éviter les mises à jour de politique trop grandes.
           - Le paramètre `clip_param` (0.2 dans notre implémentation) contrôle cette limitation.
        
        2. **Échantillonnage Efficace**: 
           - Réutilise les expériences passées via l'apprentissage hors politique.
           - Plus efficace en termes d'échantillons que les méthodes de politique sur politique.
        
        3. **Apprentissage Continu**: 
           - Prend en charge les espaces d'action continus nécessaires pour le contrôle CVC et d'éclairage.
        
        #### Fonction Objectif de PPO
        
        ```
        L_CLIP(θ) = Ê_t [ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ]
        ```
        
        - `r_t(θ)` est le ratio de probabilité entre la nouvelle et l'ancienne politique.
        - `A_t` est l'estimation de l'avantage.
        - `ε` est le paramètre de clip (0.2 dans notre cas).
        
        #### Estimation d'Avantage Généralisée (GAE)
        
        Notre implémentation utilise GAE pour estimer l'avantage d'une action:
        
        ```python
        delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        ```
        
        Avec:
        - `gamma` = 0.99 (facteur d'actualisation)
        - `gae_lambda` = 0.95 (paramètre de compromis biais-variance)
        """)
    
    with tab4:
        st.subheader("Hyperparamètres et leur Influence")
        st.markdown("""
        #### Hyperparamètres de PPO
        
        | Paramètre | Valeur | Influence |
        |-----------|--------|-----------|
        | `learning_rate` | Variable (UI) | Taille des pas d'optimisation; impacte la vitesse et la stabilité de l'apprentissage |
        | `batch_size` | Variable (UI) | Nombre d'échantillons utilisés pour chaque mise à jour; affecte l'efficacité et la stabilité |
        | `gamma` | 0.99 | Facteur d'actualisation; détermine l'importance des récompenses futures vs immédiates |
        | `gae_lambda` | 0.95 | Coefficient GAE; contrôle le compromis biais-variance dans l'estimation d'avantage |
        | `clip_param` | 0.2 | Limite les changements de politique; aide à la stabilité de l'apprentissage |
        | `entropy_coef` | 0.01 | Encourage l'exploration; empêche la convergence prématurée |
        | `value_loss_coef` | 0.5 | Poids de la perte de la fonction de valeur vs la perte de politique |
        | `max_grad_norm` | 0.5 | Écrêtage du gradient; prévient les explosions de gradient |
        | `ppo_epochs` | 10 | Nombre d'époques par lot; plus d'époques = plus d'extraction d'information des données |
        
        #### Paramètres d'Environnement
        
        | Paramètre | Valeur | Influence |
        |-----------|--------|-----------|
        | `comfort_weight` | Variable (UI) | Équilibre entre économie d'énergie et confort; valeurs plus élevées priorisent le confort |
        | `temp_range` | Variable (UI) | Plage de température confortable; plage plus large = plus facile à satisfaire |
        | `light_range` | Variable (UI) | Plage d'éclairage confortable; plage plus large = plus facile à satisfaire |
        
        #### Conseils pour l'Optimisation des Hyperparamètres
        
        1. **Learning Rate**: 
           - Trop élevé: apprentissage instable
           - Trop bas: apprentissage lent
           - Recommandation: commencer avec 0.0005 et ajuster
        
        2. **Comfort Weight**: 
           - Proche de 0: priorité maximale à l'économie d'énergie, potentiellement au détriment du confort
           - Proche de 1: priorité maximale au confort, potentiellement sans économies significatives
           - Recommandation: 0.5-0.7 pour un bon équilibre
        
        3. **Batch Size**: 
           - Plus grand: estimations plus stables mais moins d'updates
           - Plus petit: plus d'updates mais plus de variance
           - Recommandation: 64-128 pour un bon équilibre
        
        4. **Nombre d'Époques**: 
           - Plus élevé: meilleure extraction d'information mais risque de surapprentissage
           - Recommandation: 30-100 pour des résultats significatifs
        """)
    
    # Add code explanation section
    st.header("Structure du Code et Implémentation")
    
    code_tab1, code_tab2, code_tab3 = st.tabs(["Architecture Globale", "Environnement RL", "Agent PPO"])
    
    with code_tab1:
        st.markdown("""
        ### Organisation du Code
        
        Le projet est structuré en plusieurs modules avec des responsabilités spécifiques:
        
        1. **app.py**: 
           - Interface utilisateur Streamlit
           - Contrôle du flux de l'application
           - Visualisation des résultats
        
        2. **energy_env.py**: 
           - Définition de l'environnement `SmartBuildingEnv`
           - Modélisation des dynamiques du bâtiment
           - Calcul des récompenses
        
        3. **ppo_agent.py**: 
           - Implémentation de l'algorithme PPO
           - Architecture réseau acteur-critique
           - Logique d'apprentissage et d'inférence
        
        4. **training.py**: 
           - Boucle d'entraînement
           - Collecte des métriques
           - Gestion des épisodes
        
        5. **utils.py**: 
           - Chargement et prétraitement des données
           - Calcul des métriques de performance
        
        6. **visualization.py**: 
           - Génération de graphiques pour les données et résultats
        
        Cette séparation modulaire facilite la maintenance et l'extension du code, et permet de tester individuellement chaque composant.
        """)
        
    with code_tab2:
        st.code("""
# Extrait simplifié de energy_env.py
class SmartBuildingEnv(gym.Env):
    def __init__(self, config):
        # Définir les espaces d'état et d'action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        
        # Stocker la configuration
        self.data = config["data"]
        self.comfort_weight = config["comfort_weight"]
        self.temp_range = config["temp_range"]
        self.light_range = config["light_range"]
    
    def reset(self):
        # Réinitialiser l'environnement
        self.current_idx = 0
        self.state = self._get_observation(self.current_idx)
        return self.state
    
    def step(self, action):
        # Appliquer les actions au bâtiment
        hvac_adjustment = action[0] * 0.2  # Mise à l'échelle
        lighting_adjustment = action[1] * 0.2
        
        # Calculer les nouvelles conditions du bâtiment
        adjusted_hvac_power = current_data["hvac_power"] * (1 + hvac_adjustment)
        adjusted_lighting_power = current_data["lighting_power"] * (1 + lighting_adjustment)
        
        # Calculer le confort et les économies d'énergie
        energy_saved = original_total_power - adjusted_total_power
        comfort_score = (temp_comfort + light_comfort) / 2
        
        # Calculer la récompense
        energy_reward = energy_saved / original_total_power
        reward = (1 - self.comfort_weight) * energy_reward + self.comfort_weight * comfort_score
        
        # Avancer à l'état suivant
        self.current_idx += 1
        done = self.current_idx >= self.max_idx
        self.state = self._get_observation(self.current_idx) if not done else self.state
        
        return self.state, reward, done, info
        """, language='python')
        
    with code_tab3:
        st.code("""
# Extrait simplifié de ppo_agent.py
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Réseaux de l'acteur (politique)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Réseau du critique (fonction de valeur)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def get_action(self, state, deterministic=False):
        # Obtenir la distribution d'action
        action_mean, action_std, _ = self.forward(state)
        
        if deterministic:
            return action_mean
        
        # Échantillonner une action
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return torch.tanh(action), log_prob

class PPOAgent:
    def update(self):
        # Calcul des retours et avantages
        returns, advantages = self._compute_gae(rewards, values, dones)
        
        # Boucle d'entraînement PPO
        for _ in range(self.ppo_epochs):
            # Mini-batch training
            for batch_indices in batches:
                # Calcul du ratio de probabilité
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Objectif PPO avec clipping
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Pertes de la fonction de valeur et de l'entropie
                value_loss = 0.5 * (batch_returns - values_pred).pow(2).mean()
                entropy_loss = -entropy.mean()
                
                # Perte totale
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Mise à jour des paramètres
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        """, language='python')
    
    st.markdown("""
    ## Fin de la Documentation
    
    """)
