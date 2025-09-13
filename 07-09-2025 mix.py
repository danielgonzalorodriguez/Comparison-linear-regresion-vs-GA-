#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Dado un conjunto de datos generados aleatoriamente y guardados en un fichero, se pide aplicar un filtro de paso bajo,
ajustar por regresión lineal el polinomio cúbico que mejor se aproxime a dichos datos e implementar un algoritmo genético 
que calcule los coeficientes de ese polinomio para comparar los resultados obtenidos 
con los de la regresión clásica.
"""


# In[12]:


# Generación del fichero

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-5,5,100)
y=2*x**3 - 4*x**2 + x + np.random.normal(0,10,size=len(x))

np.savetxt("datos_07_09_25.txt", np.column_stack([x,y]))

plt.scatter(x,y,label="Datos generados",color="blue",alpha=0.6)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Datos generados con ruido")
plt.legend()
plt.show()


# In[23]:


fc=5 #Hz
def high_pass_filter(y,fc):
    Y=np.fft.fft(y)
    freqs=np.fft.fftfreq(len(y))
    Y[np.abs(freqs)>fc] = 0
    return np.fft.ifft(Y).real

y_transf=high_pass_filter(y,fc/len(y))
plt.scatter(x,y_transf,label="Señal filtrada",color="blue",alpha=0.6)
plt.plot(x,y_transf,color="green",alpha=0.6)
plt.xlabel("x")
plt.ylabel("y filtro")
plt.title("Datos pasados por filtro")

reconstructed = np.zeros_like(y, dtype=float)
indices = np.argsort(np.abs(freqs))[1:6]

for k in indices:
    component = np.zeros_like(Y, dtype=complex)
    component[k] = Y[k]
    component[-k] = Y[-k]  
    reconstructed += np.fft.ifft(component).real

plt.plot(x, reconstructed, label="Señal reconstruida", color="red")
plt.scatter(x,y,label="Datos generados",color="pink",alpha=0.6)
plt.plot(x,y,color="purple",alpha=0.6)
plt.legend()
plt.show()


# In[26]:


coeffs = np.polyfit(x, y, 3)  # [a, b, c, d]
p = np.poly1d(coeffs)
y_fit = p(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) 

ax1.scatter(x, y, label="Datos generados", alpha=0.6)
ax1.plot(x, y_fit, color="red", label="Ajuste polyfit")
ax1.set_title("Ajuste cúbico con polyfit")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()

ax2.scatter(x, y_transf, label="Señal filtrada", color="blue", alpha=0.6)
ax2.plot(x, y_transf, color="green", alpha=0.6, label="Filtrada")
ax2.plot(x, reconstructed, label="Señal reconstruida", color="red", linestyle="--")
ax2.scatter(x, y, label="Datos originales", color="pink", alpha=0.4)
ax2.plot(x, y, color="purple", alpha=0.4)
ax2.plot(x, y_fit, color="black", linestyle=":", label="Ajuste polyfit")
ax2.set_title("Filtrado y reconstrucción")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend()

plt.tight_layout()
plt.show()


# In[13]:


import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fitness(coeffs, x, y):
    a, b, c, d = coeffs
    y_pred = a*x**3 + b*x**2 + c*x + d
    return np.mean((y - y_pred)**2)

def mutate(coeffs, rate=0.1):
    return [c + random.uniform(-rate, rate) for c in coeffs]

x = np.linspace(-5, 5, 100)
y = 2*x**3 - 4*x**2 + x + np.random.normal(0, 20, size=len(x))

coeffs_real = [2, -4, 1, 0]

coeffs_polyfit = np.polyfit(x, y, 3)

population = [np.random.uniform(-5, 5, 4) for _ in range(50)]

for gen in range(50):
    scores = [(fitness(ind, x, y), ind) for ind in population]
    scores.sort(key=lambda s: s[0])
    population = [ind for _, ind in scores[:10]]  # élite
    
    children = []
    while len(population) + len(children) < 50:
        p1, p2 = random.sample(population, 2)
        child = [(a+b)/2 for a, b in zip(p1, p2)]
        child = mutate(child)
        children.append(child)
    population += children

best = min(population, key=lambda ind: fitness(ind, x, y))

coef_names = ["a (x^3)", "b (x^2)", "c (x)", "d (cte)"]
differences = [abs((pf - ga) / pf) * 100 if pf != 0 else np.nan
               for pf, ga in zip(coeffs_polyfit, best)]

df = pd.DataFrame({
    "Coef": coef_names,
    "Real": coeffs_real,
    "Polyfit": coeffs_polyfit,
    "GA": best,
    "Diferencia % (abs)": differences
})
display(df)

poly_real = f"{coeffs_real[0]:.3f}x^3 + {coeffs_real[1]:.3f}x^2 + {coeffs_real[2]:.3f}x + {coeffs_real[3]:.3f}"
poly_polyfit = f"{coeffs_polyfit[0]:.3f}x^3 + {coeffs_polyfit[1]:.3f}x^2 + {coeffs_polyfit[2]:.3f}x + {coeffs_polyfit[3]:.3f}"
poly_ga = f"{best[0]:.3f}x^3 + {best[1]:.3f}x^2 + {best[2]:.3f}x + {best[3]:.3f}"

print("\nPolinomio real:\n", poly_real)
print("\nPolinomio polyfit:\n", poly_polyfit)
print("\nPolinomio GA:\n", poly_ga)

plt.figure(figsize=(6,4))
plt.scatter(x, y, s=10, label="Datos", color="gray",alpha=0.5)
plt.plot(x, np.poly1d(coeffs_polyfit)(x), "g--", label="Polyfit")
plt.plot(x, best[0]*x**3 + best[1]*x**2 + best[2]*x + best[3],
         "r-", label="GA")
plt.plot(x, np.poly1d(coeffs_real)(x), "b:", linewidth=2, label="Real")
plt.legend()
plt.title("Ajuste cúbico: Real vs Polyfit vs GA")
plt.show()


# In[ ]:




