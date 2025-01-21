import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Charger les résultats sauvegardés
path_to_results = 'raw_attack_results/bayes_K10_A_cnn/mnist_fgsm_eps0.10_untargeted.pkl'
output_dir = 'adversarial_images'

# Créer le dossier de sortie
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(path_to_results, 'rb') as f:
    adv, true_ys, adv_ys, adv_logits = pickle.load(f)

# Charger les données originales (vous devez fournir X_test)
# Remplacez cette ligne par le chemin vers vos données originales si nécessaire
from cleverhans.utils_mnist import data_mnist
_, _, X_test, _ = data_mnist(train_start=0, train_end=60000, test_start=0, test_end=10000)

# Noms des catégories pour Fashion MNIST
LABEL_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Identifier les prédictions et les labels
adv_preds = np.argmax(adv_logits, axis=1)
true_labels = np.argmax(true_ys, axis=1)

# Mélanger les indices pour avoir une mosaïque variée
indices = np.arange(len(adv))
np.random.shuffle(indices)

# Extraire les informations du chemin pour générer le titre
attack_type = path_to_results.split('/')[-1].split('_')[1]  # fgsm
eps_value = path_to_results.split('_')[3].replace("eps", "")  # 0.10
title_text = f"A model's classification with {attack_type.upper()} eps {eps_value} attacks."

# Paramètres de la mosaïque
rows, cols = 5, 5
fig, axes = plt.subplots(rows * 2, cols, figsize=(12, 20))  # Double hauteur pour deux grilles

# Boucle pour afficher les images attaquées (première mosaïque)
for i, ax in zip(indices[:rows * cols], axes[:rows].flatten()):
    ax.imshow(adv[i].reshape(28, 28), cmap='gray')
    
    # Annoter avec les noms des catégories
    true_name = LABEL_NAMES[true_labels[i]]
    pred_name = LABEL_NAMES[adv_preds[i]]
    
    color = 'red' if adv_preds[i] != true_labels[i] else 'green'
    ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=8)
    ax.spines['top'].set_color(color)
    ax.spines['bottom'].set_color(color)
    ax.spines['left'].set_color(color)
    ax.spines['right'].set_color(color)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Boucle pour afficher les images originales (deuxième mosaïque)
for i, ax in zip(indices[:rows * cols], axes[rows:].flatten()):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')  # Image originale
    
    # Annoter avec le nom de la catégorie réelle
    true_name = LABEL_NAMES[true_labels[i]]
    ax.set_title(f"True: {true_name}", fontsize=8)
    ax.spines['top'].set_color('blue')
    ax.spines['bottom'].set_color('blue')
    ax.spines['left'].set_color('blue')
    ax.spines['right'].set_color('blue')
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Ajouter un titre global
fig.suptitle(title_text, fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Réserver de l'espace pour le titre

# Sauvegarder la mosaïque
output_path = os.path.join(output_dir, "mosaic_results_with_originals.png")
plt.savefig(output_path, dpi=150)
plt.close()

print(f"Mosaïque avec labels sauvegardée dans {output_path}")
