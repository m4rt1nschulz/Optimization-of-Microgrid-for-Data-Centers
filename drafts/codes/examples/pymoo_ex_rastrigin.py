import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

X1, X2 = np.meshgrid(np.linspace(-5.12, 5.12, 500), np.linspace(-5.12, 5.12, 500))

# Rastrigin math formula for n=2 variables
F = 20 + (X1**2 - 10 * np.cos(2 * np.pi * X1)) + (X2**2 - 10 * np.cos(2 * np.pi * X2))

# 3D SURFACE PLOT:
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1, X2, F, cmap='viridis', edgecolor='none', alpha=0.9)
fig.colorbar(surf, label='f(x)')
ax.set_title("Rastrigin Function (3D Surface)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")

# COUNTOUR PLOT:
plt.figure(figsize=(7, 5))
plt.contour(X1, X2, F, levels=50, cmap='viridis')
plt.colorbar(label='f(x)')
plt.title("Rastrigin Function (Contour)")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

# PYMOO OPTIMIZATION:
class RastriginProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, 
                         n_obj=1, 
                         xl=np.array([-5.12, -5.12]), 
                         xu=np.array([5.12, 5.12]))

    def _evaluate(self, x, out, *args, **kwargs):
        term1 = x[0]**2 - 10 * np.cos(2 * np.pi * x[0])
        term2 = x[1]**2 - 10 * np.cos(2 * np.pi * x[1])
        out["F"] = 20 + term1 + term2
        
algorithm = GA(pop_size=100)
res = minimize(RastriginProblem(), algorithm, ('n_gen', 50), seed=1)

# Results:
print(f"Optimum trouvé à X = {res.X}")
print(f"Valeur de f(X) = {res.F[0]:.5f}")
plt.show()