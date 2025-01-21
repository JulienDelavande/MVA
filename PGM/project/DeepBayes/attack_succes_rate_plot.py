import matplotlib.pyplot as plt

# Attack success rate
eps = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

A = [17.8, 19.4800, 29.8400, 45.0700, 74.2700, 89.3900, 94.2400, 96.0400] # GFZ
B = [17.87, 19.7100, 29.8300, 45.4600, 73.8200, 88.5500, 93.7200, 94.8000] # GFY
C = [9.84, 22.62, 75.18, 96.61, 99.98, 99.94, 99.94, 99.98] # DFZ
D = [9.83, 23.3700, 77.9600, 97.9400, 99.9700, 99.9300, 99.9300, 99.95] # DFZ
E = [22.46, 27.7200, 66.5300, 84.0500, 91.9600, 94.2400, 94.9500, 95.3300] # DBX
F = [23.22, 24.83, 30.91, 40.41, 60.73, 77.15, 85.57, 90.13] # GBZ

plt.figure(figsize=(10, 6))
plt.plot(eps, A, label='GFZ', marker='o')
plt.plot(eps, B, label='GFY', marker='o')
plt.plot(eps, C, label='DFZ', marker='o')
plt.plot(eps, D, label='DFY', marker='o')
plt.plot(eps, E, label='DBX', marker='o')
plt.plot(eps, F, label='GBZ', marker='o')

plt.xlabel('Epsilon')
plt.ylabel('Attack success rate (%)')
plt.title('Attack success rate of different models')

plt.legend()
plt.savefig('attack_success_rate.png')
plt.show()
