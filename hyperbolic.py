import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

x = np.arange(0.01,6,0.01)

sinh = np.sinh(x)
cosh = np.cosh(x)
tanh = np.tanh(x)

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, sinh,label=r'$f(x)=\mathrm{sinh}(x)$',color='red')
ax.plot(x, cosh,label=r'$f(x)=\mathrm{cosh}(x)$',color='green')
ax.plot(x, tanh,label=r'$f(x)=\mathrm{tanh}(x)$',color='blue')
plt.xlim([-0.1,3])
plt.ylim([-0.1,3])
ax.grid()
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$f(x)$',size=20)
ax.legend(fontsize=15)
plt.show()

filename = 'results/hyperbolic1.pdf'
# fig.savefig(filename, format='pdf')

plt.close()

csch = 1/np.sinh(x)
sech = 1/np.cosh(x)
coth = 1/np.tanh(x)

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, csch,label=r'$f(x)=\mathrm{csch}(x)$',color='red')
ax.plot(x, sech,label=r'$f(x)=\mathrm{sech}(x)$',color='green')
ax.plot(x, coth,label=r'$f(x)=\mathrm{coth}(x)$',color='blue')
plt.xlim([-0.1,3])
plt.ylim([-0.1,3])
ax.grid()
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$f(x)$',size=20)
ax.legend(fontsize=15)
plt.show()

filename = 'results/hyperbolic2.pdf'
# fig.savefig(filename, format='pdf')

plt.close()

sinhc = sinh / x
tanhc = tanh / x
cothc = 1 / tanhc
cschc = 1 / sinhc

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, sinhc,label=r'$f(x)=\mathrm{sinhc}(x)$',color='red')
ax.plot(x, tanhc,label=r'$f(x)=\mathrm{tanhc}(x)$',color='green')
ax.plot(x, cothc,label=r'$f(x)=\mathrm{cothc}(x)$',color='blue')
ax.plot(x, cschc,label=r'$f(x)=\mathrm{cschc}(x)$',color='black')
plt.xlim([-0.2,6])
plt.ylim([-0.2,6])
ax.grid()
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$f(x)$',size=20)
ax.legend(fontsize=15)
plt.show()

filename = 'results/hyperbolic3.pdf'
# fig.savefig(filename, format='pdf')

plt.close()
