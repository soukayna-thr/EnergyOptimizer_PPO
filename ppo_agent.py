import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Définir le périphérique
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    """
    Réseau Actor-Critic pour PPO
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Réseau Actor (politique)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Moyenne et écart-type pour les actions continues
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Réseau Critic (fonction de valeur)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """
        Passage avant à travers le réseau
        """
        # Actor : moyenne et écart-type de la distribution des actions
        actor_features = self.actor(state)
        action_mean = self.actor_mean(actor_features)
        action_std = self.actor_log_std.exp()
        
        # Critic : fonction de valeur
        value = self.critic(state)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """
        Échantillonner une action à partir de la distribution de la politique
        """
        action_mean, action_std, _ = self.forward(state)
        
        if deterministic:
            return action_mean
        
        # Créer une distribution normale
        dist = Normal(action_mean, action_std)
        
        # Échantillonner l'action à partir de la distribution
        action = dist.sample()
        
        # Calculer la probabilité logarithmique de l'action
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Appliquer la fonction tanh pour garantir que les actions sont dans [-1,1]
        action = torch.tanh(action)
        
        return action, log_prob
    
    def evaluate_actions(self, state, action):
        """
        Évaluer la probabilité logarithmique et l'entropie des actions
        """
        action_mean, action_std, value = self.forward(state)
        
        # Créer une distribution normale
        dist = Normal(action_mean, action_std)
        
        # Obtenir l'arctanh de l'action (inverse de tanh)
        action_tanh = torch.clamp(action, -0.999, 0.999)
        action_original = torch.atanh(action_tanh)
        
        # Calculer la probabilité logarithmique
        log_prob = dist.log_prob(action_original).sum(dim=-1, keepdim=True)
        
        # Calculer l'entropie
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value

class PPOAgent:
    """
    Implémentation de l'agent PPO
    """
    def __init__(self, config):
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        
        # Hyperparamètres PPO
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_param = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.ppo_epochs = 10
        
        # Initialiser le réseau actor-critic
        self.policy = ActorCritic(self.state_dim, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Stockage des données de trajectoire
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state, deterministic=False):
        """
        Sélectionner une action à partir de la politique
        """
        # Convertir l'état en tenseur
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Obtenir l'action à partir de la politique
        with torch.no_grad():
            if deterministic:
                action_mean, _, _ = self.policy(state_tensor)
                action = torch.tanh(action_mean)
            else:
                action, log_prob = self.policy.get_action(state_tensor)
        
        # Convertir en numpy
        action_np = action.squeeze().cpu().numpy()
        
        return action_np
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Stocker une transition dans la mémoire
        """
        # Convertir l'état en tenseur
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        # Obtenir la valeur et la probabilité logarithmique de l'action
        with torch.no_grad():
            _, _, value = self.policy(state_tensor)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            action_tanh = torch.clamp(action_tensor, -0.999, 0.999)
            action_original = torch.atanh(action_tanh)
            action_mean, action_std, _ = self.policy(state_tensor)
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action_original).sum(dim=-1, keepdim=True)
        
        # Stocker la transition
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.item())
        self.rewards.append(reward)
        self.values.append(value.item())
        self.dones.append(done)
    
    def update(self):
        """
        Mettre à jour la politique en utilisant l'algorithme PPO
        """
        # Si aucune transition n'est stockée, passer à la mise à jour
        if len(self.states) == 0:
            return {}
        
        # Convertir les listes en tenseurs
        states = torch.FloatTensor(self.states).to(device)
        actions = torch.FloatTensor(self.actions).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(self.rewards).unsqueeze(1).to(device)
        values = torch.FloatTensor(self.values).unsqueeze(1).to(device)
        dones = torch.FloatTensor(self.dones).unsqueeze(1).to(device)
        
        # Calculer les retours et les avantages
        returns, advantages = self._compute_gae(rewards, values, dones)
        
        # Normaliser les avantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Boucle de mise à jour PPO
        value_loss_epoch = 0
        policy_loss_epoch = 0
        entropy_epoch = 0
        
        # Créer des mini-lots
        batch_size = min(self.batch_size, states.size(0))
        batch_count = states.size(0) // batch_size
        
        for _ in range(self.ppo_epochs):
            # Créer des indices pour les lots
            indices = torch.randperm(states.size(0))
            
            for start_idx in range(0, states.size(0), batch_size):
                # Obtenir les indices du mini-lot
                end_idx = min(start_idx + batch_size, states.size(0))
                batch_indices = indices[start_idx:end_idx]
                
                # Obtenir les données du mini-lot
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Évaluer les actions
                new_log_probs, entropy, values_pred = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Calculer le ratio et les objectifs de substitution
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Objectif de substitution clippé
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Perte de valeur
                value_loss = 0.5 * (batch_returns - values_pred).pow(2).mean()
                
                # Bonus d'entropie
                entropy_loss = -entropy.mean()
                
                # Perte totale
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Mettre à jour la politique
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Enregistrer les pertes
                value_loss_epoch += value_loss.item()
                policy_loss_epoch += policy_loss.item()
                entropy_epoch += entropy_loss.item()
        
        # Réinitialiser le stockage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Calculer les pertes moyennes
        avg_value_loss = value_loss_epoch / (self.ppo_epochs * batch_count)
        avg_policy_loss = policy_loss_epoch / (self.ppo_epochs * batch_count)
        avg_entropy = entropy_epoch / (self.ppo_epochs * batch_count)
        
        return {
            'value_loss': avg_value_loss,
            'policy_loss': avg_policy_loss,
            'entropy': avg_entropy
        }
    
    def _compute_gae(self, rewards, values, dones):
        """
        Calculer les retours et les avantages en utilisant l'estimation générale des avantages (GAE)
        """
        # Initialiser les retours et les avantages
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Estimation de la dernière valeur (pour un épisode incomplet)
        if self.states:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(device)
                _, _, next_value = self.policy(state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0
        
        # Calcul de GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value_t = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def save(self, path):
        """Sauvegarder le modèle de l'agent"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Charger le modèle de l'agent"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
