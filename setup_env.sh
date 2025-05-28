#!/bin/bash

# Nom de l’environnement
ENV_NAME="air_incidents_env"

# Création de l’environnement virtuel
python3 -m venv $ENV_NAME

# Activation de l’environnement (Unix/macOS)
source $ENV_NAME/bin/activate

# Mise à jour de pip
pip install --upgrade pip

# Installation des dépendances
pip install -r requirements.txt

echo "Environment '$ENV_NAME' successfully set up "