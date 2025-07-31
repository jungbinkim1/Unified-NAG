import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from optimizers import AGM_C_optimizer, AGM_SC_optimizer, AGM_U_optimizer

os.environ['KMP_DUPLICATE_LIB_OK']='True'
N = 10000 # number of iterations

seed = 1
m = 100
n = 20
lamb = 5 # 5, 5e-2, 5e-4

np.random.seed(seed)
A = np.random.randn(m,n)
y = np.zeros((m))

x0 = 0.1*np.random.randn(n)

for i in range(m):
    y[i] = np.random.binomial(1, 1/(1+np.exp(-A[i,:]@x0)))

def logistic(t):
    return np.log(1+np.exp(t))

def logistic_prime(t):
    return (np.exp(t))/(1+np.exp(t))

def cost(X):
    result = 0
    for i in range(m):
        if (y[i] == 0):
            result += (logistic(A[i, :] @ X))
        else:
            result += (-A[i, :] @ X + logistic(A[i, :] @ X))
    return 1/m * (result + lamb*la.norm(X)**2)

def grad(X):
    result = np.zeros((n))
    for i in range(m):
        if (y[i]==0):
            result += logistic_prime(A[i, :] @ X)*(A[i, :].T)
        else:
            result += (-1 + logistic_prime(A[i, :] @ X))*(A[i, :].T)
    return 1/m * (result + 2*lamb*X)

mu = 2*lamb / m # strong convexity parameter
X0 = 1*np.random.randn(n)
stepsize = 0.01

X1, f1 = AGM_C_optimizer(N, X0, stepsize, cost, grad)
X2, f2 = AGM_SC_optimizer(N, X0, stepsize, cost, grad, mu)
X3, f3 = AGM_U_optimizer(N, X0, stepsize, cost, grad, mu)

f_sol = f3[N-1] + 1e-10

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots()
ax.loglog(range(N), f1-f_sol, label=r'AGM-\texttt{C}', color ='blue', linewidth=3)
ax.loglog(range(N), f2-f_sol, label=r'AGM-\texttt{SC}', color ='green', linewidth=3)
ax.loglog(range(N), f3-f_sol, label=r'Unified AGM (ours)', color ='red', linewidth=3, linestyle='dashed')
plt.xlabel('Iterations',size=30)
plt.ylabel(r'$f\left(x_k\right)-f\left(x^*\right)$',size=30)
ax.grid()
ax.legend(fontsize=20)
plt.show()

filename = 'results/lgst1'+'.pdf'
# fig.savefig(filename, format='pdf',  bbox_inches='tight')