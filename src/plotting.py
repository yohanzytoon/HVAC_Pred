import os
import pandas as pd
import matplotlib.pyplot as plt

# Préparation des chemins
data_dir = os.path.join('data')
train_path = os.path.join(data_dir, 'train_dataset.csv')

# Charger les données
train_data = pd.read_csv(train_path)

# Convertir les dates en datetime et extraire les caractéristiques temporelles
train_data['date'] = pd.to_datetime(train_data['date'])
train_data['hour'] = train_data['date'].dt.hour
train_data['day'] = train_data['date'].dt.day
train_data['month'] = train_data['date'].dt.month
train_data['dayofweek'] = train_data['date'].dt.dayofweek

# Trier les données par date
train_data = train_data.sort_values('date')

# Convertir les colonnes de puissance en numérique (pour éviter 'missing')
train_data['puissance_cvac'] = pd.to_numeric(train_data['puissance_cvac'], errors='coerce')
train_data['puissance_cvac_future'] = pd.to_numeric(train_data['puissance_cvac_future'], errors='coerce')

# Tracer les deux courbes côte à côte
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

# Subplot 1: puissance_cvac
axs[0].plot(train_data['date'], train_data['puissance_cvac'], color='green', label='Puissance CVAC')
axs[0].set_title('Puissance CVAC')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Puissance (kW)')
axs[0].grid(True)
axs[0].legend()

# Subplot 2: puissance_cvac_future
axs[1].plot(train_data['date'], train_data['puissance_cvac_future'], color='blue', label='Puissance CVAC Future')
axs[1].set_title('Puissance CVAC Future')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Puissance (kW)')
axs[1].grid(True)
axs[1].legend()

# Affichage
plt.tight_layout()
plt.show()
