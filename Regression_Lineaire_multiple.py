import numpy as np  # calculs numériques et matrices
import matplotlib.pyplot as plt 

import pandas as pd  #por chargemet de donner



class RegressionLineaireMultiple:
    """Classe implémentant la régression linéaire multiple avec descente de gradient"""
   

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """Constructeur : initialise les hyperparamètres du modèle"""
        self.learning_rate = learning_rate  # Taux d'apprentissage 
        self.n_iterations = n_iterations    # Nombre d'itérations de descente de gradient
        self.theta = None                   # Vecteur des paramètres de model

    def fit(self, X, y):
       
        # Ajoute une colonne de 1 à gauche pour le terme d'intercept (biais)
        # X_b.shape[0] = nombre d'exemples, np.ones crée une colonne de 1
        # np.c_ concatène : [1, X1, X2, ..., Xn] pour chaque exemple
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Initialise les paramètres aléatoirement 
        # X_b.shape[1] = n_features + 1 (intercept inclus)
        # .reshape(-1,1) pour en faire un vecteur colonne
        self.theta = np.random.randn(X_b.shape[1], 1)
        
        # Boucle principale de descente de gradient
        for _ in range(self.n_iterations):
            # Calcule le gradient de l'erreur quadratique moyenne (MSE)
            # Formule mathématique : ∇J = (2/m) * X_b.T * (X_b·θ - y)
            # où m = X_b.shape[0] (nombre d'exemples)
            gradients = 2/X_b.shape[0] * X_b.T.dot(X_b.dot(self.theta) - y.reshape(-1, 1))
            
            # Met à jour les paramètres :
            # θ_nouveau = θ_ancien - α * ∇J
            # α = learning_rate (taux d'apprentissage)
            self.theta = self.theta - self.learning_rate * gradients

    def predict(self, X):
       
        # Ajoute la colonne de 1 comme dans la phase d'entraînement
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calcule les prédictions : ŷ = X_b · θ
        # .flatten() transforme le résultat (vecteur colonne) 
        return X_b.dot(self.theta).flatten()






data = pd.read_csv('Data1.csv')#chargement de donner


X = data[['YearsExperience', 'age']].values  
# matrice (n lignes, 2 colonnes)

y = data['Salary'].values  
# vecteur salaire


plt.scatter(data['YearsExperience'], y)
# nuage de points (expérience vs salaire)

plt.xlabel('YearsExperience')
# nom axe X

plt.ylabel('Salary')
# nom axe Y

plt.title("Relation expérience - salaire")
# titre du graphique

plt.show()
# affichage





mlr = MultipleLinearRegression()  # création du modèle
mlr.fit(X, y)  # entraînement
y_pred = model.predict(X)


# affichage des résultats
print(f"Multiple LR Coefficients: {mlr.coefficients_}")# affichier les coefficients
print(f"R² Score: {mlr.r2score_:.2f}")#afficher le r2


