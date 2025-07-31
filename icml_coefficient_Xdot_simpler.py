import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
N = 1000
mu = 1 # strong convexity parameter
t0 = 0.5
t1 = 5
stepsize = (t1-t0)/N

def coeff_c(t):
    return 3/t
def coeff_sc(t):
    return 2*mu**0.5
def coeff_u(t):
    hyp = 0.5*mu**0.5*t
    return mu**0.5*np.tanh(hyp)+mu**0.5/np.tanh(hyp)+0.5*mu**0.5/(np.sinh(hyp)*np.cosh(hyp))

time = []
c1 = []
c2 = []
c3 = []
now = t0
for i in range(N):
    now = t0+i*stepsize
    time.append(now)
    c1.append(coeff_c(now))
    c2.append(coeff_sc(now))
    c3.append(coeff_u(now))

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots()
ax.plot(time, c1, label=r'AGM-\texttt{C} ODE', color ='green', linewidth=3)
ax.plot(time, c2, label=r'AGM-\texttt{SC} ODE', color ='blue', linewidth=3)
ax.plot(time, c3, label=r'Unified AGM ODE', color ='red', linewidth=3)#,  linestyle='dashed')
plt.xlabel(r'$t$',size=30)
plt.ylabel(r'Coefficient of $\dot{X}$',size=30)
y_values = [2*mu**0.5]
y_texts = [r'$2\sqrt{\mu}$']
x_values = []
x_texts = []
plt.xticks(x_values,x_texts)
plt.yticks(y_values,y_texts, fontsize=20)
ax.grid()
ax.legend(fontsize=20)
plt.show()

filename = 'results/coefficient_of_Xdot_simple'+str(mu)+'.pdf'
# fig.savefig(filename, format='pdf',  bbox_inches='tight')