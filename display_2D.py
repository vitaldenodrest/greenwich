import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

# Chargement des données
data = np.genfromtxt("test.dat", delimiter=' ')
x = data[:, 0]
y = data[:, 1]
u_true = data[:, 2]
u_pred = data[:, 3]

# Création d'une grille fine pour l'interpolation
grid_x, grid_y = np.mgrid[x.min():x.max():300j, y.min():y.max():300j]

# Fonction pour créer une heatmap lisse
def plot_smooth_heatmap(x, y, z, title, cmap='RdBu_r', show_points=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Interpolation sur grille régulière pour un rendu lisse
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    
    # Heatmap lisse
    im = ax.imshow(grid_z.T, extent=(x.min(), x.max(), y.min(), y.max()), 
                   origin='lower', cmap=cmap, aspect='auto', interpolation='bilinear')
    
    # Points de données si demandé
    if show_points:
        ax.scatter(x, y, c='black', s=5, alpha=0.3, edgecolors='none')
    
    # Colorbar et labels
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    plt.tight_layout()
    return fig, ax

# Affichage de la solution prédite
plot_smooth_heatmap(x, y, u_pred, 'Solution Prédite (u_pred)')

# Affichage de la solution vraie
plot_smooth_heatmap(x, y, u_true, 'Solution Vraie (u_true)')

# Affichage de l'erreur
plot_smooth_heatmap(x, y, u_pred - u_true, 'Erreur (u_pred - u_true)')

plt.show()