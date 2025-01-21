import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

# Charger les résultats sauvegardés
path_to_results = 'raw_attack_results/bayes_K10_A_cnn/mnist_fgsm_eps0.10_untargeted.pkl'
output_dir = 'adversarial_images'

# Créer le dossier de sortie
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Charger les résultats
with open(path_to_results, 'rb') as f:
    adv, true_ys, adv_ys, adv_logits = pickle.load(f)

# Identifier les exemples mal classifiés
adv_preds = np.argmax(adv_logits, axis=1)
true_labels = np.argmax(true_ys, axis=1)
misclassified_indices = np.where(adv_preds != true_labels)[0]

print(f"Nombre d'exemples mal classifiés : {len(misclassified_indices)}")

# Enregistrer les images mal classifiées
for idx, i in enumerate(misclassified_indices):
    plt.imshow(adv[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {true_labels[i]}, Predicted: {adv_preds[i]}")
    plt.axis('off')
    output_path = os.path.join(output_dir, f"misclassified_{idx}.png")
    plt.savefig(output_path)
    plt.close()


html_output_path = os.path.join(output_dir, "report.html")

with open(html_output_path, "w") as f:
    f.write("<html><body>\n")
    f.write("<h1>Rapport des exemples mal classifiés</h1>\n")
    for idx, i in enumerate(misclassified_indices):
        image_path = f"misclassified_{idx}.png"
        f.write(f"<div><h3>True: {true_labels[i]}, Predicted: {adv_preds[i]}</h3>\n")
        f.write(f"<img src='{image_path}' style='width:150px; height:150px;'/></div><br>\n")
    f.write("</body></html>\n")
