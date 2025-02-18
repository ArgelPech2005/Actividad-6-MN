#   Codigo que implementa el esquema numerico 
#   de interpolacion para determinar la raiz de
#   una ecuacion
"""   Autor:
   Argel Jesus Pech Manrique
   argelpech098@gmail.com
   Version 2.0 : 17/02/2025 10:50pm
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Función original
def f(x):
    #EJERCICIO 1
    #return x**3 - 6*x**2 + 11*x - 6
    #EJERCICIO 2
    return np.sin(x)-x/2
    #EJERCICIO 3
    #return np.exp(-x) - x

# Interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Método de Bisección
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    print("Iter |       a       |       b       |       c       |      f(c)      | ")
    print("-" * 85)
    
    for _ in range(max_iter):
        c = (a + b) / 2
        print(f"{_+1:4d} | {a:.8f} | {b:.8f} | {c:.8f} | {f(c):.8f} |")
        if abs(func(c)) < tol or (b - a) / 2 < tol:
            return c
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2  # Retorna la mejor estimación de la raíz

# Selección de tres puntos de interpolación

#EJERCICIO 1
#Corrida 1
#x0,x1,x2 = 0 , 1 , 2 
#Corrida 2
#x0,x1,x2 = 1 , 2 , 3
#Corrida 3
#x0,x1,x2 = 2 , 3 , 4
#FIN EJERCICIO 1

#EJERCICIO 2
#Corrida 1
#x0,x1,x2 = -1,0,1
#Corrida 2
x0,x1,x2 = 0,1,2
#EJERCICIO 3
#x0,x1,x2,x3 = 0.0,0.25,0.5,1.0  
#Eje 1, 2
x_points = np.array([x0, x1, x2])
y_points = f(x_points)

# Construcción del polinomio interpolante
# mediante interpolacion de Lagrange
x_vals = np.linspace(x0, x2, 100)

#x_points = np.array([x0, x1, x2, x3])  # Se almacenan los puntos de x en un array
#y_points = f(x_points)  # Se evalúa la función en los puntos seleccionados

# Construcción del polinomio interpolante
#x_vals = np.linspace(x0, x3, 100)  # Se generan 100 puntos entre x0 y x3 para graficar
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Encontrar raíz del polinomio interpolante usando bisección
# en el intervalo inducido por los puntos donde se hace la interpolacion
#ejer 1,, 2
root = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x2)
#Ejercicio 3
#root = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x3)

error_ABS = np.abs(y_interp - f(x_vals))  # Se calcula el error absoluto
error_REL = error_ABS / np.where(np.abs(f(x_vals)) == 0, 1, np.abs(f(x_vals)))  # Se calcula el error relativo evitando división por cero
error_CUA = error_ABS**2  # Se calcula el error cuadrático

# Encabezado de la tabla
print(f"{'Iteración':<10}|{'x':<12}|{'Error absoluto':<18}|{'Error relativo':<18}|{'Error cuadrático'}")
print("-" * 80)

# Iterar sobre los valores calculados
for i, (x_val, error_abs, error_rel, error_cuad) in enumerate(zip(x_vals, error_ABS, error_REL, error_CUA)):  
    # Se imprime la información en formato de tabla
    print(f"{i+1:<10}|{x_val:<12.6f}|{error_abs:<18.6e}|{error_rel:<18.6e}|{error_cuad:.6e}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

#Grafica Errores
ax[0].plot(x_vals, error_ABS, label="Error Absoluto", color='purple')
ax[0].plot(x_vals, error_REL, label="Error Relativo", color='orange')
ax[0].plot(x_vals, error_CUA, label="Error Cuadrático", color='brown')
ax[0].set_xlabel("x")
ax[0].set_ylabel("Errores")
ax[0].legend()
ax[0].grid(True)

# Subgráfica 2: Función y interpolación
ax[1].plot(x_vals, f(x_vals), label="$f(x) = sin(x)-x/2$", linestyle='dashed', color='blue')  # Se grafica la función original
ax[1].plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')  # Se grafica la interpolación
ax[1].axhline(0, color='black', linewidth=0.5, linestyle='--')  # Línea horizontal en y = 0
ax[1].axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")  # Se marca la raíz encontrada
ax[1].scatter(x_points, y_points, color='black', label="Puntos de interpolación")  # Se marcan los puntos de interpolación
ax[1].set_xlabel("x")
ax[1].set_ylabel("f(x)")
ax[1].legend()
ax[1].grid(True)

plt.savefig("interpolacion_raices.png")  
plt.show()  # Mostrar la gráfica

"""# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = x^3 - 4x + 1", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.savefig("interpolacion_raices.png")  # Guarda la imagen
plt.show()
"""""

# Imprimir la raíz encontrada
print(f"La raíz aproximada usando interpolación es: {root:.4f}")
