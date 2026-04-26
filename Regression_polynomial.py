

import numpy as np  # Bibliothèque pour les calculs numériques et les tableaux
import matplotlib.pyplot as plt  # Bibliothèque pour les graphiques 2D
import pandas as pd# poure chsrgemet de donnet


class RegressionPolynomiale:
   

    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000):
      
        self.degree = degree                    # Degré du polynôme à générer
        self.learning_rate = learning_rate      # Pas de descente de gradient
        self.n_iterations = n_iterations        # Nombre d'itérations
        self.theta = None                       # Paramètres du modèle (initialisés plus tard)
        
        # Crée un générateur de features polynomiales de degré spécifié
        # PolynomialFeatures transforme [x1, x2] en [1, x1, x2, x1², x1*x2, x2², ...]
        self.poly = PolynomialFeatures(degree=degree)

    def fit(self, X, y):
       
        # Étape 1 : Transformation polynomiale des features
        # Transforme les features originales en features polynomiales
     
        # La colonne de 1 pour l'intercept est automatiquement ajoutée par PolynomialFeatures
        X_poly = self.poly.fit_transform(X)
        
        # Étape 2 : Initialisation aléatoire des paramètres
        # Le nombre de paramètres = nombre de features polynomiales générées
       (intercept, x, x²)
        self.theta = np.random.randn(X_poly.shape[1], 1)
        
        # Étape 3 : Descente de gradient pour optimiser les paramètres
        for _ in range(self.n_iterations):
            # Calcul du gradient de l'erreur quadratique moyenne (identique à la régression linéaire)
            # Formule : (2/m) * X_poly.T * (X_poly·θ - y)
            gradients = 2/X_poly.shape[0] * X_poly.T.dot(X_poly.dot(self.theta) - y.reshape(-1, 1))
            
            # Mise à jour des paramètres par descente de gradient
            # θ = θ - α * ∇J(θ)
            self.theta = self.theta - self.learning_rate * gradients

    def predict(self, X):
      
        # Applique la même transformation polynomiale qu'à l'entraînement
        # Important : utiliser transform() et pas fit_transform() pour utiliser les mêmes paramètres
        X_poly = self.poly.transform(X)
        
        # Calcule les prédictions : ŷ = X_poly · θ
        # .flatten() pour retourner un tableau 1D
        return X_poly.dot(self.theta).flatten()







data = pd.read_csv('Data1.csv')

X = data[['YearsExperience']].values
# variable indépendante

y = data['Salary'].values
# variable cible





model = PolynomialRegression(degree=3)
# modèle polynomial degré 3

model.fit(X, y)
# entraînement


y_pred = model.predict(X)


plt.scatter(X, y, color='blue', label='Data')
# points réels

# tri pour courbe propre
sorted_index = X.flatten().argsort()

plt.plot(X[sorted_index], y_pred[sorted_index], color='red', label='Polynomial Regression')
# courbe polynomiale

plt.xlabel('YearsExperience')
# axe X

plt.ylabel('Salary')
# axe Y

plt.title('Polynomial Regression (Class Implementation)')
# titre

plt.legend()
# légende

plt.show()
# affichage
