import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

def naca_4digit(number, n_points=100):
    m = int(number[0]) / 100.0
    p = int(number[1]) / 10.0
    t = int(number[2:]) / 100.0
    
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2
    
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 
                   0.2843*x**3 - 0.1015*x**4)
    
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    if m > 0 and p > 0:
        mask = x < p
        yc[mask] = m/p**2 * (2*p*x[mask] - x[mask]**2)
        dyc_dx[mask] = 2*m/p**2 * (p - x[mask])
        
        mask = x >= p
        yc[mask] = m/(1-p)**2 * ((1-2*p) + 2*p*x[mask] - x[mask]**2)
        dyc_dx[mask] = 2*m/(1-p)**2 * (p - x[mask])
    
    theta = np.arctan(dyc_dx)
    
    x_upper = x - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    
    x_lower = x + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)
    
    return x_upper, y_upper, x_lower, y_lower

def create_3d_wing(naca_number='2412', n_sections=20, span=10, chord_root=1.0, 
                   chord_tip=0.5, sweep=0, dihedral=0):
    x_u, y_u, x_l, y_l = naca_4digit(naca_number)
    
    x_airfoil = np.concatenate([x_u, x_l[::-1]])
    y_airfoil = np.concatenate([y_u, y_l[::-1]])
    
    y_span = np.linspace(0, span/2, n_sections)
    
    X = []
    Y = []
    Z = []
    
    for i, y in enumerate(y_span):
        chord = chord_root + (chord_tip - chord_root) * (y / (span/2))
        x_offset = y * np.tan(np.radians(sweep))
        z_offset = y * np.tan(np.radians(dihedral))
        
        x_section = x_airfoil * chord + x_offset
        z_section = y_airfoil * chord + z_offset
        y_section = np.full_like(x_section, y)
        
        X.append(x_section)
        Y.append(y_section)
        Z.append(z_section)
    
    return np.array(X), np.array(Y), np.array(Z)

# Read file and extract values
f = open("chat_history.txt")
text = f.read()

pattern = r'=\s*([^\s=]+(?:\s+[^\s=]+)*?)(?=\s+\w+\s*[-=]|$)'
matches = re.findall(pattern, text)

values = []
for match in matches:
    match = match.strip()
    try:
        value = int(match)
    except ValueError:
        try:
            value = float(match)
        except ValueError:
            value = match
    values.append(value)

# Create and plot wing
X, Y, Z = create_3d_wing(naca_number='2412', n_sections=values[0], span=values[1], 
                       chord_root=values[2], chord_tip=values[3], sweep=values[4], dihedral=values[5])

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(X.shape[0]):
    ax.plot(X[i], Y[i], Z[i], 'b-', linewidth=0.5, alpha=0.6)
    ax.plot(X[i], -Y[i], Z[i], 'b-', linewidth=0.5, alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
