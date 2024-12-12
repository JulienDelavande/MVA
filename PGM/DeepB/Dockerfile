FROM python:3.6-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers nécessaires dans l'image
COPY requirements.txt /app/
COPY cleverhans /app/cleverhans

# Installer les bibliothèques Python nécessaires
RUN pip install --no-cache-dir -r requirements.txt

# Installer CleverHans depuis le dossier local
RUN pip install /app/cleverhans

# Copier tout le code du projet dans le conteneur
COPY . /app/

# Exposer un port (facultatif, en fonction des besoins du projet)
EXPOSE 5000

# Définir la commande par défaut
CMD ["/bin/bash"]


