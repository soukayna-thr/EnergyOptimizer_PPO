import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Définir le périphérique de calcul (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    """
    Réseau Actor-Critic combiné pour PPO.
    L'acteur (Actor) détermine la politique (stratégie) et le critique (Critic) estime la valeur des états.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initialise le réseau Actor-Critic.
        
        Args:
            state_dim (int): Dimension de l'espace d'états
            action_dim (int): Dimension de l'espace d'actions
            hidden_dim (int): Nombre de neurones dans les couches cachées (défaut: 64)
        """
        super(ActorCritic, self).__init__()
        
        # Réseau Actor (politique) - génère des actions
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # Première couche cachée
            nn.Tanh(),                         # Fonction d'activation
            nn.Linear(hidden_dim, hidden_dim), # Deuxième couche cachée
            nn.Tanh()                         # Fonction d'activation
        )
        
        # Couche de sortie pour la moyenne des actions (une par dimension d'action)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        
        # Paramètre appris pour l'écart-type des actions (log std pour assurer la positivité)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Réseau Critic (fonction de valeur) - estime la qualité des états
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # Première couche cachée
            nn.Tanh(),                         # Fonction d'activation
            nn.Linear(hidden_dim, hidden_dim),  # Deuxième couche cachée
            nn.Tanh(),                         # Fonction d'activation
            nn.Linear(hidden_dim, 1)          # Couche de sortie (valeur scalaire)
        )
        
    def forward(self, state):
        """
        Passe avant dans le réseau.
        
        Args:
            state (torch.Tensor): Tenseur représentant l'état actuel
            
        Returns:
            tuple: (moyenne des actions, écart-type des actions, valeur estimée de l'état)
        """
        # Passage dans le réseau Actor
        actor_features = self.actor(state)
        
        # Calcul de la moyenne des actions
        action_mean = self.actor_mean(actor_features)
        
        # Calcul de l'écart-type (en exponentiant le log std pour garantir la positivité)
        action_std = self.actor_log_std.exp()
        
        # Passage dans le réseau Critic pour obtenir la valeur de l'état
        value = self.critic(state)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """
        Échantillonne une action à partir de la politique actuelle.
        
        Args:
            state (torch.Tensor): État courant
            deterministic (bool): Si True, retourne l'action moyenne sans exploration
            
        Returns:
            tuple/tenseur: Action et sa log probabilité (mode stochastique) ou juste action (mode déterministe)
        """
        # Obtenir les paramètres de la distribution
        action_mean, action_std, _ = self.forward(state)
        
        # En mode déterministe, retourner simplement la moyenne
        if deterministic:
            return torch.tanh(action_mean)  # tanh pour borner l'action
        
        # Créer une distribution normale avec les paramètres calculés
        dist = Normal(action_mean, action_std)
        
        # Échantillonner une action
        action = dist.sample()
        
        # Calculer sa probabilité logarithmique
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Appliquer tanh pour borner l'action dans [-1, 1] et calculer la nouvelle log prob
        action = torch.tanh(action)
        
        return action, log_prob
    
    def evaluate_actions(self, state, action):
        """
        Évalue des actions déjà prises pour calculer leurs probabilités sous la politique actuelle.
        
        Args:
            state (torch.Tensor): État dans lequel l'action a été prise
            action (torch.Tensor): Action à évaluer
            
        Returns:
            tuple: (log probabilité de l'action, entropie de la distribution, valeur estimée de l'état)
        """
        # Obtenir les paramètres de la distribution et la valeur de l'état
        action_mean, action_std, value = self.forward(state)
        
        # Créer la distribution normale
        dist = Normal(action_mean, action_std)
        
        # Inverser la transformation tanh pour obtenir l'action dans l'espace original
        action_tanh = torch.clamp(action, -0.999, 0.999)  # Éviter les valeurs à ±1 pour atanh
        action_original = torch.atanh(action_tanh)
        
        # Calculer la log probabilité de l'action
        log_prob = dist.log_prob(action_original).sum(dim=-1, keepdim=True)
        
        # Calculer l'entropie de la distribution (mesure d'exploration)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value

class PPOAgent:
    """
    Implémentation complète d'un agent utilisant l'algorithme PPO (Proximal Policy Optimization).
    """
    def __init__(self, config):
        """
        Initialise l'agent PPO avec une configuration donnée.
        
        Args:
            config (dict): Dictionnaire de configuration contenant:
                - state_dim (int): Dimension de l'espace d'états
                - action_dim (int): Dimension de l'espace d'actions
                - lr (float): Taux d'apprentissage
                - batch_size (int): Taille des lots pour les mises à jour
        """
        # Configuration de base
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        
        # Hyperparamètres de PPO
        self.gamma = 0.99          # Facteur de discount pour les récompenses futures
        self.gae_lambda = 0.95     # Paramètre lambda pour GAE (Generalized Advantage Estimation)
        self.clip_param = 0.2      # Paramètre de clipping pour la fonction objectif
        self.entropy_coef = 0.01   # Coefficient pour le bonus d'entropie
        self.value_loss_coef = 0.5  # Poids de la perte sur les valeurs
        self.max_grad_norm = 0.5    # Valeur maximale pour le clipping des gradients
        self.ppo_epochs = 10        # Nombre d'epochs de mise à jour par lot de données
        
        # Initialisation du réseau de politique
        self.policy = ActorCritic(self.state_dim, self.action_dim).to(device)
        
        # Optimiseur Adam pour la mise à jour des poids
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Mémoire pour stocker les transitions (méthode "on-policy")
        self.states = []          # États visités
        self.actions = []         # Actions prises
        self.log_probs = []       # Log probabilités des actions
        self.rewards = []         # Récompenses obtenues
        self.values = []          # Valeurs estimées des états
        self.dones = []           # Indicateurs de fin d'épisode
    
    def select_action(self, state, deterministic=False):
        """
        Sélectionne une action selon la politique actuelle.
        
        Args:
            state (np.array): État courant de l'environnement
            deterministic (bool): Si True, pas d'exploration (utilisation pour l'évaluation)
            
        Returns:
            np.array: Action sélectionnée
        """
        # Conversion de l'état en tenseur PyTorch
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Passage dans le réseau sans calcul de gradient (pour efficacité)
        with torch.no_grad():
            if deterministic:
                # Mode déterministe - pas d'exploration
                action_mean, _, _ = self.policy(state_tensor)
                action = torch.tanh(action_mean)
            else:
                # Mode stochastique - exploration
                action, log_prob = self.policy.get_action(state_tensor)
        
        # Conversion en numpy array pour interaction avec l'environnement
        action_np = action.squeeze().cpu().numpy()
        
        return action_np
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Stocke une transition dans la mémoire de l'agent.
        
        Args:
            state (np.array): État avant l'action
            action (np.array): Action prise
            reward (float): Récompense obtenue
            next_state (np.array): État résultant
            done (bool): Si l'épisode est terminé
        """
        # Conversion des états en tenseurs
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        # Calculs sans gradient
        with torch.no_grad():
            # Estimation de la valeur de l'état
            _, _, value = self.policy(state_tensor)
            
            # Conversion de l'action et inversion de tanh pour le calcul de probabilité
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            action_tanh = torch.clamp(action_tensor, -0.999, 0.999)
            action_original = torch.atanh(action_tanh)
            
            # Calcul des paramètres de la distribution
            action_mean, action_std, _ = self.policy(state_tensor)
            dist = Normal(action_mean, action_std)
            
            # Calcul de la log probabilité de l'action
            log_prob = dist.log_prob(action_original).sum(dim=-1, keepdim=True)
        
        # Stockage de la transition
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.item())
        self.rewards.append(reward)
        self.values.append(value.item())
        self.dones.append(done)
    
    def update(self):
        """
        Met à jour la politique en utilisant l'algorithme PPO.
        
        Returns:
            dict: Statistiques de l'entraînement (pertes moyennes)
        """
        # Si pas assez de données, ne rien faire
        if len(self.states) == 0:
            return {}
        
        # Conversion des listes en tenseurs PyTorch
        states = torch.FloatTensor(self.states).to(device)
        actions = torch.FloatTensor(self.actions).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(self.rewards).unsqueeze(1).to(device)
        values = torch.FloatTensor(self.values).unsqueeze(1).to(device)
        dones = torch.FloatTensor(self.dones).unsqueeze(1).to(device)
        
        # Calcul des retours et avantages avec GAE
        returns, advantages = self._compute_gae(rewards, values, dones)
        
        # Normalisation des avantages pour stabiliser l'entraînement
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Initialisation des compteurs pour les statistiques
        value_loss_epoch = 0
        policy_loss_epoch = 0
        entropy_epoch = 0
        
        # Détermination de la taille des lots
        batch_size = min(self.batch_size, states.size(0))
        batch_count = states.size(0) // batch_size
        
        # Boucle d'entraînement PPO (plusieurs epochs sur les mêmes données)
        for _ in range(self.ppo_epochs):
            # Mélange des indices pour créer des lots aléatoires
            indices = torch.randperm(states.size(0))
            
            # Parcours par lots
            for start_idx in range(0, states.size(0), batch_size):
                # Sélection des indices du lot courant
                end_idx = min(start_idx + batch_size, states.size(0))
                batch_indices = indices[start_idx:end_idx]
                
                # Extraction des données du lot
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Évaluation des actions avec la politique actuelle
                new_log_probs, entropy, values_pred = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Calcul du ratio des probabilités (nouvelle/ancienne politique)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calcul de la fonction objectif clippée (core de PPO)
                surr1 = ratio * batch_advantages  # Terme non clippé
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages  # Terme clippé
                policy_loss = -torch.min(surr1, surr2).mean()  # On minimise la négation du minimum
                
                # Perte sur les valeurs (erreur quadratique)
                value_loss = 0.5 * (batch_returns - values_pred).pow(2).mean()
                
                # Perte d'entropie (pour encourager l'exploration)
                entropy_loss = -entropy.mean()  # On minimise la négation de l'entropie
                
                # Perte totale (combinaison pondérée des différentes pertes)
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Rétropropagation
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clipping des gradients pour éviter les explosions
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                # Mise à jour des poids
                self.optimizer.step()
                
                # Enregistrement des statistiques
                value_loss_epoch += value_loss.item()
                policy_loss_epoch += policy_loss.item()
                entropy_epoch += entropy_loss.item()
        
        # Réinitialisation de la mémoire après mise à jour
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Calcul des moyennes des pertes
        avg_value_loss = value_loss_epoch / (self.ppo_epochs * batch_count)
        avg_policy_loss = policy_loss_epoch / (self.ppo_epochs * batch_count)
        avg_entropy = entropy_epoch / (self.ppo_epochs * batch_count)
        
        # Retour des statistiques
        return {
            'value_loss': avg_value_loss,
            'policy_loss': avg_policy_loss,
            'entropy': avg_entropy
        }
    
    def _compute_gae(self, rewards, values, dones):
        """
        Calcule les retours et avantages généralisés (GAE).
        
        Args:
            rewards (torch.Tensor): Tenseur des récompenses
            values (torch.Tensor): Tenseur des valeurs estimées
            dones (torch.Tensor): Tenseur des indicateurs de fin d'épisode
            
        Returns:
            tuple: (retours, avantages)
        """
        # Initialisation des tenseurs de sortie
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Estimation de la dernière valeur (pour les épisodes incomplets)
        if self.states:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(device)
                _, _, next_value = self.policy(state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0
        
        # Calcul des avantages avec Generalized Advantage Estimation (GAE)
        gae = 0
        for t in reversed(range(len(rewards))):
            # Détermination de la valeur et du terminal pour le pas suivant
            if t == len(rewards) - 1:
                next_value_t = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value_t = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            # Calcul du delta TD (erreur temporelle)
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            
            # Mise à jour de GAE
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            # Stockage des résultats
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def save(self, path):
        """
        Sauvegarde le modèle et l'optimiseur dans un fichier.
        
        Args:
            path (str): Chemin du fichier de sauvegarde
        """
        torch.save({
            'policy': self.policy.state_dict(),  # Paramètres du réseau
            'optimizer': self.optimizer.state_dict()  # État de l'optimiseur
        }, path)
    
    def load(self, path):
        """
        Charge un modèle et un optimiseur depuis un fichier.
        
        Args:
            path (str): Chemin du fichier à charger
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])