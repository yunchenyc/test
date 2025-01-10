#%% 1
import numpy as np
import matplotlib.pyplot as plt

#%% 2

def r(a, b, theta_s, theta_0, tau, omega):
    r = a+b*np.exp(np.cos(theta_s-theta_0+tau*omega))
    
    return r

def dr(t, a, b, theta_s, theta_0, k, c, omega):
    tau = k*t+c
    dr = -omega*b*np.exp(np.cos(theta_s-theta_0+tau*omega))*np.sin(theta_s-theta_0+omega*tau)*k
    return dr

#%% 3

t = np.arange(0, 0.5, 0.01)
a = 1
b = 1
theta_s = 0
theta_0 = 0
omegas = [-240, -120, 0, 120, 240]

k = np.pi*2
c = 0
tau = k*t + c

rts = []
for omega in omegas:
    omega = omega/180*np.pi
    rts.append(r(a, b, theta_s, theta_0, tau, omega))


plt.figure()
for i, rt in enumerate(rts):
    plt.plot(t, rt, label=omegas[i])
plt.legend()
plt.show()


# %%
# heatmap??
# correlation??

plt.figure()
omega = 120/180*np.pi

theta_0_ls = np.linspace(0, np.pi*2, 100)

ht = np.zeros((len(t), 100))
for i, theta0 in enumerate(theta_0_ls):
    ht[:, i] = r(a, b, theta_s, theta0, tau, omega)

plt.subplot(131)
plt.imshow(ht.T)

omega = -120/180*np.pi

theta_0_ls = np.linspace(0, np.pi*2, 100)

ht = np.zeros((len(t), 100))
for i, theta0 in enumerate(theta_0_ls):
    ht[:, i] = r(a, b, theta_s, theta0, tau, omega)

plt.subplot(132)
plt.imshow(ht.T)

omega = 240/180*np.pi

theta_0_ls = np.linspace(0, np.pi*2, 100)

ht = np.zeros((len(t), 100))
for i, theta0 in enumerate(theta_0_ls):
    ht[:, i] = r(a, b, theta_s, theta0, tau, omega)

plt.subplot(133)
plt.imshow(ht.T)

plt.show()



# %%
corr = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        corr[i, j] = np.corrcoef(ht[:, i], ht[:, j])[1, 0]

plt.figure()
plt.imshow(corr)
plt.show()


# %%
