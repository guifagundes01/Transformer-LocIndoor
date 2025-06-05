import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette("muted")

p = [100, 150, 200, 250]
o = [8.18, 8.18, 8.17, 8.05]
oi = [7.22, 7.21, 7.20, 7.09]
a = [9.00, 9.08, 9.23, 9.01]
ai = [7.99, 8.08, 8.15, 8.02]

# p  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# o  = 8.05
# oi = 7.22
# a  = [9.01, 8.89, 8.57, 8.54, 8.60, 8.49, 8.43, 8.42, 8.37, 8.34, 8.48, 8.43, 8.48, 8.50, 8.45, 8.46]
# ai = [8.02, 7.89, 7.69, 7.66, 7.65, 7.57, 7.49, 7.50, 7.43, 7.45, 7.55, 7.46, 7.56, 7.59, 7.55,7.57]

colors = plt.get_cmap('tab10')
color1 = colors(0)  # blueish
color2 = colors(2)  # greenish

color1 = palette[0]  # e.g., muted blue
color2 = palette[2]  # e.g., muted green
color3 = palette[1]

plt.figure(figsize=(8, 5))
# plt.axhline(y=o, color=color1, linestyle='--', label=f'Real = {o}m')
# plt.axhline(y=oi, color=color1, linestyle='--', label=f'Real = {oi}m')
# plt.axhline(y=min(ai), color=color3, linestyle='--', label=f'Min Artificial = {min(ai)}')
# plt.plot(p, o, linestyle='--',  color=color1, linewidth=2, label="Real without classification error")
plt.plot(p, oi, linestyle='-',  color=color1, linewidth=2, label="Real")
# plt.plot(p, a, linestyle='--',  color=color2, linewidth=2, label="Artificial")
plt.plot(p, ai, linestyle='-',  color=color2, linewidth=2, label="Artificial")
# plt.plot(p, [ai[i] - oi[i] for i in range(len(oi))], linestyle='-', color=color3, linewidth=2, label="Difference")
plt.xticks(p)
# plt.yticks([i / 2 for i in range(12, 18)])

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend()
plt.xlabel("Dataset size")
plt.xlabel("#Points per dimension")
plt.ylabel('Error (m)')
# plt.title('Beautiful 4-Curve Plot')
# plt.show()
plt.savefig("figures/error_per_point_wo_discretization.png", bbox_inches='tight', format="png")
