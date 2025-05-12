import numpy as np
import matplotlib.pyplot as plt

def load_params(file_path):
    with open(file_path, 'r') as f:
        line = f.readline()
        w1, w2, b = map(float, line.strip().split(','))
    return w1, w2, b

def plot_regions(ax, w1, w2, b, cmap='coolwarm', alpha=0.2):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    z = w1 * xx + w2 * yy + b
    zz = (z >= 0).astype(int)  # 1 si z ≥ 0, sino 0

    ax.contourf(xx, yy, zz, levels=[-1, 0, 1], colors=['lightcoral', 'lightblue'], alpha=alpha)

def plot_decision_line(ax, w1, w2, b, label, color):
    x = np.linspace(-0.5, 1.5, 100)
    if w2 != 0:
        y = -(w1 * x + b) / w2
        ax.plot(x, y, label=label, color=color)
    else:
        x_val = -b / w1
        ax.axvline(x=x_val, label=label, color=color)

def plot_inputs_and_axes(ax):
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for x, y in inputs:
        ax.plot(x, y, 'ko')
        ax.text(x + 0.03, y + 0.03, f'({x},{y})', fontsize=8)

#AND
fig_and, ax_and = plt.subplots()
w1_cpp, w2_cpp, b_cpp = load_params('result/cpp_and_model.txt')
w1_py, w2_py, b_py = load_params('result/py_and_model.txt')

plot_regions(ax_and, w1_cpp, w2_cpp, b_cpp, alpha=0.15)
plot_decision_line(ax_and, w1_cpp, w2_cpp, b_cpp, 'C++ AND', 'blue')

plot_regions(ax_and, w1_py, w2_py, b_py, alpha=0.15)
plot_decision_line(ax_and, w1_py, w2_py, b_py, 'Python AND', 'green')

plot_inputs_and_axes(ax_and)
ax_and.set_title("Líneas y regiones de decisión - AND")
ax_and.set_xlim(-0.5, 1.5)
ax_and.set_ylim(-0.5, 1.5)
ax_and.legend()
ax_and.grid(True)

#OR
fig_or, ax_or = plt.subplots()
w1_cpp, w2_cpp, b_cpp = load_params('result/cpp_or_model.txt')
w1_py, w2_py, b_py = load_params('result/py_or_model.txt')

plot_regions(ax_or, w1_cpp, w2_cpp, b_cpp, alpha=0.15)
plot_decision_line(ax_or, w1_cpp, w2_cpp, b_cpp, 'C++ OR', 'red')

plot_regions(ax_or, w1_py, w2_py, b_py, alpha=0.15)
plot_decision_line(ax_or, w1_py, w2_py, b_py, 'Python OR', 'orange')

plot_inputs_and_axes(ax_or)
ax_or.set_title("Líneas y regiones de decisión - OR")
ax_or.set_xlim(-0.5, 1.5)
ax_or.set_ylim(-0.5, 1.5)
ax_or.legend()
ax_or.grid(True)

plt.show()
