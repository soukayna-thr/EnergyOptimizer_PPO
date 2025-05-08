import plotly.graph_objects as go
import numpy as np
import pandas as pd

def create_building_model(num_floors=3, num_rooms_per_floor=4, room_size=10, building_type="office"):
    """
    Crée un modèle 3D amélioré d'un bâtiment avec plusieurs étages et pièces.
    
    Args:
        num_floors: Nombre d'étages
        num_rooms_per_floor: Nombre de pièces par étage (doit être un carré parfait)
        room_size: Taille des pièces
        building_type: Type de bâtiment ("office", "residential", etc.)
        
    Returns:
        dict: Dictionnaire avec les coordonnées des pièces et autres informations
    """
    building = {
        'rooms': [],
        'walls': [],
        'floors': [],
        'corridors': [],
        'windows': [],
        'type': building_type,
        'metadata': {
            'num_floors': num_floors,
            'num_rooms_per_floor': num_rooms_per_floor,
            'total_area': num_rooms_per_floor * room_size * room_size * num_floors
        }
    }
    
    # Calculer les dimensions du bâtiment
    grid_size = int(np.sqrt(num_rooms_per_floor))
    building_width = grid_size * room_size
    building_length = grid_size * room_size
    floor_height = room_size / 2
    
    # Créer un corridor central
    corridor_width = room_size / 4
    
    # Créer les pièces
    room_id = 0
    room_types = ["bureau", "salle de réunion", "espace commun", "stockage"]
    
    for floor in range(num_floors):
        floor_rooms = []
        for row in range(grid_size):
            for col in range(grid_size):
                # Déterminer le type de pièce (varie selon l'étage et la position)
                room_type_idx = (floor + row + col) % len(room_types)
                room_type = room_types[room_type_idx]
                
                # Coordonnées de la pièce avec espacements pour corridors
                x0 = col * room_size
                y0 = row * room_size
                z0 = floor * floor_height
                
                # Ajouter une variation aux dimensions des pièces pour plus de réalisme
                width_var = 1.0 if (col % 2 == 0) else 0.9
                length_var = 1.0 if (row % 2 == 0) else 0.95
                height_var = 1.0
                
                actual_width = room_size * width_var
                actual_length = room_size * length_var
                actual_height = floor_height * height_var
                
                # Coordonnées ajustées pour pièces non uniformes
                room = {
                    'id': room_id,
                    'floor': floor,
                    'position': (row, col),
                    'type': room_type,
                    'coords': {
                        'x': [x0, x0 + actual_width, x0 + actual_width, x0, x0, x0 + actual_width, x0 + actual_width, x0],
                        'y': [y0, y0, y0 + actual_length, y0 + actual_length, y0, y0, y0 + actual_length, y0 + actual_length],
                        'z': [z0, z0, z0, z0, z0 + actual_height, z0 + actual_height, z0 + actual_height, z0 + actual_height]
                    },
                    'center': (x0 + actual_width/2, y0 + actual_length/2, z0 + actual_height/2),
                    'temperature': 22.0 + (floor * 0.2) + ((row+col) % 3 - 1) * 0.5,  # Variation de température selon la position
                    'light_level': 500 + ((floor * 20) - ((row+col) % 5) * 30),  # Variation d'éclairage
                    'occupancy': 1 if ((floor + row + col) % 3 == 0) else 0,  # Distribution d'occupation
                    'dimensions': {
                        'width': actual_width,
                        'length': actual_length,
                        'height': actual_height
                    }
                }
                
                floor_rooms.append(room)
                room_id += 1
        
        building['rooms'].extend(floor_rooms)
        
        # Ajouter le plancher pour cet étage
        floor_coords = {
            'x': [0, building_width, building_width, 0],
            'y': [0, 0, building_length, building_length],
            'z': [floor * floor_height, floor * floor_height, floor * floor_height, floor * floor_height]
        }
        building['floors'].append(floor_coords)
        
        # Ajouter un corridor horizontal et vertical pour chaque étage
        h_corridor = {
            'x': [0, building_width, building_width, 0],
            'y': [(building_length/2) - corridor_width/2, (building_length/2) - corridor_width/2,
                 (building_length/2) + corridor_width/2, (building_length/2) + corridor_width/2],
            'z': [floor * floor_height, floor * floor_height, floor * floor_height, floor * floor_height]
        }
        
        v_corridor = {
            'x': [(building_width/2) - corridor_width/2, (building_width/2) + corridor_width/2,
                 (building_width/2) + corridor_width/2, (building_width/2) - corridor_width/2],
            'y': [0, 0, building_length, building_length],
            'z': [floor * floor_height, floor * floor_height, floor * floor_height, floor * floor_height]
        }
        
        building['corridors'].append(h_corridor)
        building['corridors'].append(v_corridor)
    
    # Ajouter des fenêtres extérieures
    for floor in range(num_floors):
        # Fenêtres sur les côtés du bâtiment
        for i in range(1, grid_size):
            # Fenêtres côté nord
            window_north = {
                'x': [i * room_size - room_size*0.3, i * room_size + room_size*0.3, 
                      i * room_size + room_size*0.3, i * room_size - room_size*0.3],
                'y': [0, 0, room_size*0.05, room_size*0.05],
                'z': [floor * floor_height + floor_height*0.2, floor * floor_height + floor_height*0.2,
                     floor * floor_height + floor_height*0.8, floor * floor_height + floor_height*0.8]
            }
            # Fenêtres côté sud
            window_south = {
                'x': [i * room_size - room_size*0.3, i * room_size + room_size*0.3, 
                      i * room_size + room_size*0.3, i * room_size - room_size*0.3],
                'y': [building_length, building_length, building_length - room_size*0.05, building_length - room_size*0.05],
                'z': [floor * floor_height + floor_height*0.2, floor * floor_height + floor_height*0.2,
                     floor * floor_height + floor_height*0.8, floor * floor_height + floor_height*0.8]
            }
            building['windows'].append(window_north)
            building['windows'].append(window_south)
            
            # Fenêtres côté est
            window_east = {
                'x': [0, room_size*0.05, room_size*0.05, 0],
                'y': [i * room_size - room_size*0.3, i * room_size - room_size*0.3,
                     i * room_size + room_size*0.3, i * room_size + room_size*0.3],
                'z': [floor * floor_height + floor_height*0.2, floor * floor_height + floor_height*0.2,
                     floor * floor_height + floor_height*0.8, floor * floor_height + floor_height*0.8]
            }
            # Fenêtres côté ouest
            window_west = {
                'x': [building_width, building_width - room_size*0.05, building_width - room_size*0.05, building_width],
                'y': [i * room_size - room_size*0.3, i * room_size - room_size*0.3,
                     i * room_size + room_size*0.3, i * room_size + room_size*0.3],
                'z': [floor * floor_height + floor_height*0.2, floor * floor_height + floor_height*0.2,
                     floor * floor_height + floor_height*0.8, floor * floor_height + floor_height*0.8]
            }
            building['windows'].append(window_east)
            building['windows'].append(window_west)
    
    return building

def update_building_state(building, temperatures, light_levels, occupancies):
    """
    Met à jour l'état du bâtiment avec de nouvelles valeurs
    
    Args:
        building: Le modèle du bâtiment
        temperatures: Liste des températures pour chaque pièce
        light_levels: Liste des niveaux d'éclairage pour chaque pièce
        occupancies: Liste des occupations pour chaque pièce
    
    Returns:
        dict: Le bâtiment mis à jour
    """
    for i, room in enumerate(building['rooms']):
        room['temperature'] = temperatures[i] if i < len(temperatures) else room['temperature']
        room['light_level'] = light_levels[i] if i < len(light_levels) else room['light_level']
        room['occupancy'] = occupancies[i] if i < len(occupancies) else room['occupancy']
    
    return building

def get_color_for_temperature(temp):
    """Convertit une température en couleur"""
    # Bleu pour froid (18°C), blanc pour neutre (22°C), rouge pour chaud (28°C)
    if temp <= 18:
        return 'rgb(0, 0, 255)'  # Bleu
    elif temp >= 28:
        return 'rgb(255, 0, 0)'  # Rouge
    else:
        # Interpolation linéaire entre bleu, blanc et rouge
        if temp < 22:
            # Entre bleu et blanc
            ratio = (temp - 18) / (22 - 18)
            r = int(0 + ratio * 255)
            g = int(0 + ratio * 255)
            b = int(255)
        else:
            # Entre blanc et rouge
            ratio = (temp - 22) / (28 - 22)
            r = int(255)
            g = int(255 * (1 - ratio))
            b = int(255 * (1 - ratio))
        
        return f'rgb({r}, {g}, {b})'

def get_color_for_light(light):
    """Convertit un niveau d'éclairage en opacité"""
    # 300 lux = faible, 500 lux = moyen, 700 lux = élevé
    if light <= 300:
        return 0.2
    elif light >= 700:
        return 0.9
    else:
        # Interpolation linéaire
        return 0.2 + (light - 300) / (700 - 300) * 0.7

def plot_building_3d(building, show_legend=True, show_controls=True):
    """
    Crée une visualisation 3D du bâtiment avec Plotly
    
    Args:
        building: Le modèle du bâtiment
        show_legend: Afficher la légende ou non
        show_controls: Afficher les contrôles interactifs
        
    Returns:
        plotly.graph_objects.Figure: La figure Plotly
    """
    fig = go.Figure()
    
    # Ajouter les planchers pour chaque étage
    for i, floor in enumerate(building['floors']):
        fig.add_trace(go.Mesh3d(
            x=floor['x'],
            y=floor['y'],
            z=floor['z'],
            color='lightgrey',
            opacity=0.7,
            i=[0],
            j=[1],
            k=[2],
            name=f"Plancher Étage {i}",
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Plancher - Étage {i}"
        ))
    
    # Ajouter les corridors
    for i, corridor in enumerate(building['corridors']):
        fig.add_trace(go.Mesh3d(
            x=corridor['x'],
            y=corridor['y'],
            z=corridor['z'],
            color='lightblue',
            opacity=0.3,
            i=[0],
            j=[1],
            k=[2],
            name=f"Corridor {i}",
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Corridor - Étage {i//2}"
        ))
    
    # Ajouter les fenêtres
    for i, window in enumerate(building['windows']):
        fig.add_trace(go.Mesh3d(
            x=window['x'],
            y=window['y'],
            z=window['z'],
            color='skyblue',
            opacity=0.4,
            i=[0],
            j=[1],
            k=[2],
            name=f"Fenêtre {i}",
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Fenêtre"
        ))
    
    # Ajouter les pièces avec leurs couleurs basées sur la température et l'éclairage
    for room in building['rooms']:
        color = get_color_for_temperature(room['temperature'])
        opacity = get_color_for_light(room['light_level'])
        
        # Les murs de la pièce
        fig.add_trace(go.Mesh3d(
            x=room['coords']['x'],
            y=room['coords']['y'],
            z=room['coords']['z'],
            color=color,
            opacity=opacity,
            i=[0, 0, 0, 4, 4, 4],
            j=[1, 2, 3, 5, 6, 7],
            k=[2, 3, 1, 6, 7, 5],
            name=f"Pièce {room['id']} (Étage {room['floor']})",
            showlegend=False,
            hoverinfo='text',
            hovertext=f"<b>Pièce {room['id']} - {room.get('type', 'Standard')}</b><br><br>" + 
                      f"<b>Étage:</b> {room['floor']}<br>" +
                      f"<b>Température:</b> {room['temperature']:.1f}°C<br>" +
                      f"<b>Éclairage:</b> {room['light_level']} lux<br>" +
                      f"<b>Occupé:</b> {'Oui' if room['occupancy'] > 0 else 'Non'}<br><br>" +
                      f"<i>Cliquez pour plus de détails</i>"
        ))
        
        # Marqueur pour le centre de la pièce (pour l'occupation)
        if room['occupancy'] > 0:
            fig.add_trace(go.Scatter3d(
                x=[room['center'][0]],
                y=[room['center'][1]],
                z=[room['center'][2]],
                mode='markers',
                marker=dict(
                    size=10,
                    color='green',
                    symbol='circle',
                    line=dict(color='darkgreen', width=2)
                ),
                name=f"Occupant (Pièce {room['id']})",
                showlegend=False,
                hoverinfo='text',
                hovertext=f"<b>Occupant</b><br>Pièce {room['id']} - {room.get('type', 'Standard')}<br>Étage: {room['floor']}"
            ))
    
    # Ajouter une échelle de couleur pour la température
    temp_values = [18, 20, 22, 24, 26, 28]
    for i, temp in enumerate(temp_values):
        fig.add_trace(go.Scatter3d(
            x=[30],
            y=[i * 2],
            z=[5],
            mode='markers',
            marker=dict(
                size=10,
                color=get_color_for_temperature(temp),
            ),
            name=f"{temp}°C",
            showlegend=show_legend
        ))
    
    # Ajouter une échelle pour le niveau d'éclairage
    light_values = [300, 400, 500, 600, 700]
    for i, light in enumerate(light_values):
        fig.add_trace(go.Scatter3d(
            x=[32],
            y=[i * 2],
            z=[5],
            mode='markers',
            marker=dict(
                size=10,
                color='white',
                opacity=get_color_for_light(light),
                line=dict(color='black', width=1)
            ),
            name=f"{light} lux",
            showlegend=show_legend
        ))
    
    # Ajouter un marqueur pour l'occupation
    fig.add_trace(go.Scatter3d(
        x=[30],
        y=[12],
        z=[5],
        mode='markers',
        marker=dict(
            size=10,
            color='green',
        ),
        name="Occupé",
        showlegend=show_legend
    ))
    
    # Configurer la mise en page
    fig.update_layout(
        title="Simulation 3D du Bâtiment Intelligent",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Étage",
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=-1.5, z=1)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            x=0,
            y=1,
            title="Légende"
        ),
        # Ajouter des options d'interactivité
        scene_dragmode="turntable",
        hoverlabel=dict(
            bgcolor="rgba(50, 50, 50, 0.9)",  # Fond gris foncé semi-transparent
            font_size=14,
            font_family="Arial",
            font=dict(color="white")  # Texte blanc
        ),
        updatemenus=[
            # Bouton pour réinitialiser la vue
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Vue par défaut",
                        method="relayout",
                        args=[{"scene.camera": dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=-1.5, z=1)
                        )}]
                    ),
                    dict(
                        label="Vue de dessus",
                        method="relayout",
                        args=[{"scene.camera": dict(
                            up=dict(x=0, y=1, z=0),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=0, y=0, z=2.5)
                        )}]
                    ),
                    dict(
                        label="Vue latérale",
                        method="relayout",
                        args=[{"scene.camera": dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=2.5, y=0, z=0)
                        )}]
                    ),
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ] if show_controls else []
    )
    
    return fig

def simulate_building_optimization(building, optimized_df):
    """
    Crée une animation de l'optimisation du bâtiment
    
    Args:
        building: Le modèle du bâtiment
        optimized_df: DataFrame avec les résultats optimisés
    
    Returns:
        List[plotly.graph_objects.Figure]: Liste de figures Plotly pour l'animation
    """
    frames = []
    num_rooms = len(building['rooms'])
    
    # Si le dataframe ne contient pas les colonnes nécessaires, créer des données fictives
    if not all(col in optimized_df.columns for col in ['hvac_action', 'lighting_action', 'occupancy']):
        return [plot_building_3d(building)]
    
    # Limiter à 3 frames pour ne pas surcharger l'interface
    step = max(1, len(optimized_df) // 3)
    
    for i in range(0, min(len(optimized_df), 3*step), step):
        row = optimized_df.iloc[i]
        
        # Extraire les actions de l'agent
        hvac_adjustment = row.get('hvac_action', 0)
        lighting_adjustment = row.get('lighting_action', 0)
        occupancy = row.get('occupancy', np.zeros(num_rooms))
        
        # Pour la simulation, appliquer les ajustements différemment pour chaque pièce
        temps = []
        lights = []
        occs = []
        
        for room_id in range(num_rooms):
            # L'effet de l'action dépend de l'étage et de la position
            floor_factor = 1 + 0.2 * building['rooms'][room_id]['floor']
            pos_factor = 1 + 0.1 * (building['rooms'][room_id]['position'][0] + building['rooms'][room_id]['position'][1])
            
            # Température de base entre 20 et 24°C
            base_temp = 22 + (room_id % 5 - 2)
            # Ajuster en fonction de l'action de l'agent
            temp = base_temp - hvac_adjustment * floor_factor * pos_factor
            temps.append(min(28, max(18, temp)))
            
            # Niveau d'éclairage de base entre 400 et 600 lux
            base_light = 500 + (room_id % 5 - 2) * 50
            # Ajuster en fonction de l'action de l'agent
            light = base_light + lighting_adjustment * 200 * floor_factor
            lights.append(min(700, max(300, light)))
            
            # Occupation: utiliser les données si disponibles, sinon alternance
            if isinstance(occupancy, np.ndarray) and len(occupancy) > room_id:
                occ = occupancy[room_id]
            else:
                occ = 1 if ((i + room_id) % 8) < 4 else 0
            occs.append(occ)
        
        # Mettre à jour l'état du bâtiment
        updated_building = update_building_state(building.copy(), temps, lights, occs)
        
        # Créer la figure pour ce pas de temps
        fig = plot_building_3d(updated_building, show_legend=(i==0))
        fig.update_layout(title=f"Simulation - Étape {i//step + 1}")
        
        frames.append(fig)
    
    return frames

def plot_building_animation(building, optimized_df):
    """
    Crée une série de visualisations pour montrer l'animation
    
    Args:
        building: Le modèle du bâtiment
        optimized_df: DataFrame avec les résultats optimisés
    
    Returns:
        List[plotly.graph_objects.Figure]: Liste de figures Plotly
    """
    return simulate_building_optimization(building, optimized_df)