# Utiliser une image avec Python 3.6 préinstallé
FROM python:3.6-slim

# Mettre à jour le système et installer les outils nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean

# Installer TensorFlow et Keras (versions spécifiques)
RUN pip install tensorflow==1.10.1 keras==2.2.2

# Créer un dossier pour la sauvegarde des résultats
RUN mkdir -p /app/save

# Définir le répertoire de travail
WORKDIR /app

# Commande par défaut : ouvrir un terminal interactif
CMD ["bash"]
