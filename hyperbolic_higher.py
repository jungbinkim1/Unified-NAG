import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
x = np.arange(0.01,6,0.01)
sinh = np.sinh(x)
cosh = np.cosh(x)
tanh = np.tanh(x)

sinh3 = np.array([])
isf = 0
# Euler method applied to the defining ODE
for i in x:
    isf = isf + 0.01 * (1 + isf ** 3) ** (1 / 3)
    sinh3 = np.append(sinh3, isf)
sinh30 = np.array([])
isf = 0
for i in x:
    isf = isf + 0.01 * (1 + isf ** 30) ** (1 / 30)
    sinh30 = np.append(sinh30, isf)

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, sinh,label=r'$f(x)=\mathrm{sinh}_2(x)$',color='red')
ax.plot(x, sinh3, label=r'$f(x)=\mathrm{sinh}_3(x)$', color='green')#, linestyle='dashed'
ax.plot(x, sinh30, label=r'$f(x)=\mathrm{sinh}_{30}(x)$', color='blue')
plt.xlim([-0.1,3])
plt.ylim([-0.1,3])
ax.grid()
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$f(x)$',size=20)
ax.legend(fontsize=15)
plt.show()

filename = 'results/hh1.pdf'
# fig.savefig(filename, format='pdf')

plt.close()

cosh3 = (1 + sinh3 ** 3) ** (1 / 3)
cosh30 = (1 + sinh30 ** 30) ** (1 / 30)

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, cosh,label=r'$f(x)=\mathrm{cosh}_2(x)$',color='red')
ax.plot(x, cosh3,label=r'$f(x)=\mathrm{cosh}_3(x)$',color='green')
ax.plot(x, cosh30,label=r'$f(x)=\mathrm{cosh}_{30}(x)$',color='blue')
plt.xlim([-0.1,3])
plt.ylim([-0.1,3])
ax.grid()
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$f(x)$',size=20)
ax.legend(fontsize=15)
plt.show()

filename = 'results/hh2.pdf'
# fig.savefig(filename, format='pdf')

plt.close()

tanh3 = sinh3 / cosh3
tanh30 = sinh30 / cosh30

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, tanh,label=r'$f(x)=\mathrm{tanh}_2(x)$',color='red')
ax.plot(x, tanh3,label=r'$f(x)=\mathrm{tanh}_{3}(x)$',color='green')
ax.plot(x, tanh30,label=r'$f(x)=\mathrm{tanh}_{30}(x)$',color='blue')
plt.xlim([-0.2,6])
plt.ylim([-0.2,6])
ax.grid()
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$f(x)$',size=20)
ax.legend(fontsize=15)
plt.show()

filename = 'results/hh3.pdf'
# fig.savefig(filename, format='pdf')

plt.close()
