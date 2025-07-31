import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from optimizers import ATM_U_optimizer, ATM_SC_optimizer

os.environ['KMP_DUPLICATE_LIB_OK']='True'
N = 10000 # number of iterations
seed = 1
m = 50
n = 50
lamb = 1 # 1, 1e-3, 1e-6

np.random.seed(seed)
A = np.random.randn(m,n)
A = A + A.T
b = 10*np.random.randn(m,1)

def cost(X):
    return 1/m * (0.5*la.norm(A@X-b)**2 + lamb*la.norm(X)**3)

# For the expressions of grad and Hess, see for example the following articles:
# https://math.stackexchange.com/questions/2143052/%E2%88%92-2-compute-the-hessian-of-f-and-show-it-is-positive-defini
# https://math.stackexchange.com/questions/4634856/hessian-of-norm-cubed
def grad(X):
    return 1/m * (A.T @ (A@X-b) + 3*lamb*la.norm(X)*X)

def hess(X):
    return 1/m * (A.T @ A + 3*lamb*(la.norm(X)*np.identity(n) + 1/la.norm(X) * X@X.T))

mu = 1.5*lamb / m # strong convexity parameter
X0 = 0.1*np.random.randn(n,1)
stepsize = 0.001
C = (1/3) * (1/6)**2
coeff_of_cube = 2/3 # h(x)=(3/2)||x||^3

M=100000
Xtemp, ftemp = ATM_U_optimizer(M, X0, stepsize, cost, grad, mu, C, coeff_of_cube, hess)
f_sol = ftemp[M - 1] + 1e-10
X1, f1 = ATM_U_optimizer(N, X0, stepsize, cost, grad, 0, C, coeff_of_cube, hess)
X2, f2 = ATM_SC_optimizer(N, X0, stepsize, cost, grad, mu, C, coeff_of_cube, hess)
X3, f3 = ATM_U_optimizer(N, X0, stepsize, cost, grad, mu, C, coeff_of_cube, hess)

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots()
ax.loglog(range(N), f1-f_sol, label=r'ATM-\texttt{C} (Wibisono et al.)', color ='blue', linewidth=3)
ax.loglog(range(N), f2-f_sol, label=r'ATM-\texttt{SC} (ours)', color ='green', linewidth=3)
ax.loglog(range(N), f3-f_sol, label=r'Unified ATM (ours)', color ='red', linestyle='dashed', linewidth=3)
plt.xlabel('Iterations',size=30)
plt.ylabel(r'$f\left(x_k\right)-f\left(x^*\right)$',size=30)
ax.grid()
ax.legend(fontsize=20)
plt.show()

filename = 'results/high1'+'.pdf'
# fig.savefig(filename, format='pdf',  bbox_inches='tight')