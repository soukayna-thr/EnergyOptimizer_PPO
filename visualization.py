import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_energy_consumption(df):
    """
    Tracer les modèles de consommation d'énergie au fil du temps.
    
    Args:
        df (DataFrame): Données de consommation d'énergie
        
    Returns:
        Figure: Objet figure de Plotly
    """
    # Créer une figure avec un axe y secondaire
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Ajouter la consommation d'énergie du HVAC
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['hvac_power'],
            name="HVAC",
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # Ajouter la consommation d'énergie de l'éclairage
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['lighting_power'],
            name="Lighting",
            line=dict(color='#ff7f0e', width=2)
        )
    )
    
    # Ajouter la consommation d'énergie de la charge connectée
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['plug_load_power'],
            name="Plug Load",
            line=dict(color='#2ca02c', width=2)
        )
    )
    
    # Ajouter la consommation totale d'énergie
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['total_power'],
            name="Total",
            line=dict(color='#d62728', width=3)
        )
    )
    
    # Ajouter la température extérieure sur l'axe secondaire
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['outside_temp'],
            name="Température extérieure (°C)",
            line=dict(color='#9467bd', width=2, dash='dash')
        ),
        secondary_y=True
    )
    
    # Ajouter l'occupation sous forme de graphique en aires
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['occupancy'] * df['total_power'].max() * 0.3,  # Mise à l'échelle pour la visibilité
            name="Occupation",
            fill='tozeroy',
            mode='none',
            fillcolor='rgba(180,180,180,0.3)'
        )
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title="Modèles de consommation d'énergie",
        xaxis_title="Temps",
        yaxis_title="Puissance (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500
    )
    
    # Mettre à jour l'axe y secondaire
    fig.update_yaxes(title_text="Température (°C)", secondary_y=True)
    
    return fig

def plot_comfort_metrics(original_df, optimized_df):
    """
    Tracer les indicateurs de confort avant et après optimisation.
    
    Args:
        original_df (DataFrame): Données originales
        optimized_df (DataFrame): Données optimisées
        
    Returns:
        Figure: Objet figure de Plotly
    """
    # Créer deux sous-graphiques
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Température intérieure", "Niveau de lumière"),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Ajouter les courbes de température
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=original_df['indoor_temp'],
            name="Température originale",
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=optimized_df['adjusted_indoor_temp'],
            name="Température optimisée",
            line=dict(color='#ff7f0e', width=2)
        ),
        row=1, col=1
    )
    
    # Ajouter la plage de confort pour la température
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=[20] * len(original_df),  # Limite inférieure de confort
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=[25] * len(original_df),  # Limite supérieure de confort
            line=dict(color="rgba(0,0,0,0)"),
            fill='tonexty',
            fillcolor='rgba(0,176,246,0.2)',
            name="Plage de confort"
        ),
        row=1, col=1
    )
    
    # Ajouter les courbes de lumière
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=original_df['light_level'],
            name="Lumière originale",
            line=dict(color='#1f77b4', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=optimized_df['adjusted_light_level'],
            name="Lumière optimisée",
            line=dict(color='#ff7f0e', width=2)
        ),
        row=2, col=1
    )
    
    # Ajouter la plage de confort pour la lumière
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=[400] * len(original_df),  # Limite inférieure de confort
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=[700] * len(original_df),  # Limite supérieure de confort
            line=dict(color="rgba(0,0,0,0)"),
            fill='tonexty',
            fillcolor='rgba(0,176,246,0.2)',
            name="Plage de confort"
        ),
        row=2, col=1
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title="Indicateurs de confort avant et après optimisation",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=600,
        yaxis1=dict(title="Température (°C)"),
        yaxis2=dict(title="Niveau de lumière (lux)")
    )
    
    return fig

def plot_optimization_comparison(original_df, optimized_df):
    """
    Tracer la consommation d'énergie avant et après optimisation.
    
    Args:
        original_df (DataFrame): Données originales
        optimized_df (DataFrame): Données optimisées
        
    Returns:
        Figure: Objet figure de Plotly
    """
    # Créer une figure
    fig = go.Figure()
    
    # Ajouter la consommation d'énergie originale
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=original_df['total_power'],
            name="Consommation originale",
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # Ajouter la consommation d'énergie optimisée
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=optimized_df['adjusted_total_power'],
            name="Consommation optimisée",
            line=dict(color='#ff7f0e', width=2)
        )
    )
    
    # Ajouter les économies d'énergie sous forme de zone remplie
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=original_df['total_power'],
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=optimized_df['adjusted_total_power'],
            fill='tonexty',
            fillcolor='rgba(0,176,246,0.2)',
            line=dict(width=0),
            name="Économies d'énergie"
        )
    )
    
    # Ajouter l'occupation pour référence
    fig.add_trace(
        go.Scatter(
            x=original_df['timestamp'],
            y=original_df['occupancy'] * original_df['total_power'].max() * 0.3,  # Mise à l'échelle pour la visibilité
            name="Occupation",
            fill='tozeroy',
            mode='none',
            fillcolor='rgba(180,180,180,0.3)'
        )
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title="Consommation d'énergie : Avant vs Après optimisation",
        xaxis_title="Temps",
        yaxis_title="Puissance (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_training_progress(progress):
    """
    Tracer la progression de l'entraînement au fil des époques.
    
    Args:
        progress (dict): Données de progression de l'entraînement
        
    Returns:
        Figure: Objet figure de Plotly
    """
    # Créer une figure avec un axe y secondaire
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Ajouter la courbe de la récompense moyenne
    fig.add_trace(
        go.Scatter(
            x=progress['epochs'],
            y=progress['rewards'],
            name="Récompense moyenne",
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # Ajouter la courbe de l’énergie économisée
    fig.add_trace(
        go.Scatter(
            x=progress['epochs'],
            y=progress['energy_saved'],
            name="Énergie économisée (%)",
            line=dict(color='#ff7f0e', width=2)
        ),
        secondary_y=True
    )
    
    # Ajouter la courbe du score de confort
    fig.add_trace(
        go.Scatter(
            x=progress['epochs'],
            y=progress['comfort_score'],
            name="Score de confort (%)",
            line=dict(color='#2ca02c', width=2)
        ),
        secondary_y=True
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title="Progression de l'entraînement",
        xaxis_title="Époques",
        yaxis_title="Récompense moyenne",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=400
    )
    
    # Mettre à jour l'axe y secondaire
    fig.update_yaxes(title_text="Pourcentage (%)", secondary_y=True)
    
    return fig


def plot_control_actions(optimized_df):
    """
    Tracer les actions de contrôle au cours du temps.
    
    Args:
        optimized_df (DataFrame): Données optimisées avec les actions de contrôle
        
    Returns:
        Figure: Objet figure de Plotly
    """
    # Créer une figure avec un axe y secondaire
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Ajouter la courbe d’ajustement du système HVAC
    fig.add_trace(
        go.Scatter(
            x=optimized_df['timestamp'],
            y=optimized_df['hvac_adjustment'] * 100,  # Conversion en pourcentage
            name="Ajustement HVAC (%)",
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # Ajouter la courbe d’ajustement de l’éclairage
    fig.add_trace(
        go.Scatter(
            x=optimized_df['timestamp'],
            y=optimized_df['lighting_adjustment'] * 100,  # Conversion en pourcentage
            name="Ajustement éclairage (%)",
            line=dict(color='#ff7f0e', width=2)
        )
    )
    
    # Ajouter la température extérieure sur l'axe secondaire
    fig.add_trace(
        go.Scatter(
            x=optimized_df['timestamp'],
            y=optimized_df['outside_temp'],
            name="Température extérieure (°C)",
            line=dict(color='#9467bd', width=2, dash='dash')
        ),
        secondary_y=True
    )
    
    # Ajouter l’occupation sous forme de zone
    fig.add_trace(
        go.Scatter(
            x=optimized_df['timestamp'],
            y=optimized_df['occupancy'] * 100,  # Conversion en pourcentage
            name="Occupation (%)",
            fill='tozeroy',
            mode='none',
            fillcolor='rgba(180,180,180,0.3)'
        )
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title="Actions de contrôle au fil du temps",
        xaxis_title="Temps",
        yaxis_title="Ajustement (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500
    )
    
    # Mettre à jour l’axe y secondaire
    fig.update_yaxes(title_text="Température (°C)", secondary_y=True)
    
    # Ajouter une ligne horizontale à zéro pour les ajustements
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    
    return fig
