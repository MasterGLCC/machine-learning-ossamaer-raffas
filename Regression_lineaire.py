import numpy as np  # importer la bibliothèque numpy pour les calculs mathématiques et les tableaux
import matplotlib.pyplot as plt#  por les figure  

import pandas as pd  # importer la bibliothèque pour chargement de donner

class RegressionLineaire:
    """Classe implémentant la régression linéaire avec descente de gradient"""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        # Constructeur de la classe
        self.learning_rate = learning_rate  # Taux d'apprentissage (pas de descente de gradient)
        self.n_iterations = n_iterations    # Nombre d'itérations pour l'optimisation
        self.theta = None                   # Paramètres du modèle (poids), initialisés à None

    def fit(self, X, y):
       
        # Ajouter une colonne de 1 
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Concatène une colonne de 1 à gauche de X
        
        # Initialiser les paramètres theta aléatoirement
        # X_b.shape[1] = nombre de features + 1 
        self.theta = np.random.randn(X_b.shape[1], 1)  # Vecteur colonne de paramètres aléatoires

       
        for _ in range(self.n_iterations):
            # Calcul du gradient de la fonction de coût (erreur quadratique moyenne)
            # Formule: (2/m) * X_b.T * (X_b * theta - y)
            # où m = X_b.shape[0] (nombre d'exemples)
            gradients = 2/X_b.shape[0] * X_b.T.dot(X_b.dot(self.theta) - y.reshape(-1, 1))
            
            # Mise à jour des paramètres par descente de gradient
            # Nouveau theta = ancien theta - taux_apprentissage * gradient
            self.theta = self.theta - self.learning_rate * gradients

    def predict(self, X):
        """Prédit les valeurs pour de nouvelles données X"""
        # Ajouter une colonne de 1 pour l'intercept 
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calcul des prédictions : X_b * theta
        # flatten() transforme le vecteur colonne en tableau 1D
        return X_b.dot(self.theta).flatten()  # Retourne les prédictions sous forme de tableau 

#  data base
data=pd.read_csv('Salary_Data.csv')
data # afficher les donner s il exest

X = data[['YearsExperience']].values  # selectionner le colone matrice n x 1 selectioner les colone

y = data['Salary'].values #selectioner le  Salary #vecteur


# Cette partie sert à afficher un graphique pour voir la relation entre les données

plt.scatter(data['YearsExperience'], y)
# Crée un nuage de points :
# - axe X : YearsExperience (expérience)
# - axe Y : y (salaire)
# Chaque point représente une observation

plt.xlabel('YearsExperience')
# Donne un nom à l’axe horizontal (X)

plt.ylabel('Salary')
# Donne un nom à l’axe vertical (Y)

plt.show()
# Affiche le graphique à l’écran











  

slr=SimpleLinearRegression()  #  Crée une instance de type SimpleLinearRegression
slr.fit(X,y)   #    Entraîne le modèle sur les données X (entrées) et y (sorties)




print("simple lr coefficients:",slr.coefficients_)  # Affiche les coefficients du modèle (intercept et pente)
print("R2 Score:",slr.r2score_)    # Affiche le score R² qui mesure la qualité de l'ajustement du modèle
#Très bon ajustement (proche de 1 = parfait).










# Split manuel des données 80% train # 20% test
train_size = int(0.8 * len(X))  # Détermine la taille de l'ensemble d'entraînement 80% 
X_train, X_test = X[:train_size], X[train_size:] # Sélectionne les 80% premières données pour l'entraînement (X_train)
y_train, y_test = y[:train_size], y[train_size:]  ## Sélectionne les données correspondantes pour y_train et y_test


#fait la Prédictions sur les données d'entraînement
y_train_pred = slr.predict(X_train)

# fait laPrédictions sur les données de test
y_test_pred = slr.predict(X_test)


# Calcul du MSE
MSE = np.mean((y_test - y_test_pred)**2) # clculer moyen
print("MSE :", MSE)

plt.scatter(X, y, color='blue', label='Data')
# Affiche les données sous forme de nuage de points (scatter plot)
# X → axe horizontal
# y → axe vertical
# color='blue' → couleur des points en bleu
# label='Data' → nom utilisé dans la légende

plt.plot(X, slr.y_pred, color='red', label='regression lineare')
# Trace la droite de régression
# X → mêmes valeurs sur l’axe horizontal
# slr.y_pred → valeurs prédites par le modèle (ŷ)
# color='red' → couleur de la droite en rouge
# label → nom dans la légende

plt.title("regression lineaire simple")
# Ajoute un titre au graphique

plt.xlabel("X")
# Nom de l’axe horizontal

plt.ylabel("y")
# Nom de l’axe vertical

plt.legend()
# Affiche la légende 
plt.show()
# Affiche le graphique final












