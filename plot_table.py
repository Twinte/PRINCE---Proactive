import matplotlib.pyplot as plt
import numpy as np

# Data from the table
methods = ['Proactive', 'ECS-HDSR', 'ACSP-FL', 'Entropy-only', 'Random']
rounds_60 = [35, 48, 55, 98, 106]
rounds_70 = [51, 100, 87, 136, 109]

# Bar configuration
x = np.arange(len(methods))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the bars
# Note: 'hatch' is added for black&white print compatibility
rects1 = ax.bar(x - width/2, rounds_60, width, label='Rounds to 60%', 
                color='#4c72b0', edgecolor='black', hatch='///')
rects2 = ax.bar(x + width/2, rounds_70, width, label='Rounds to 70%', 
                color='#dd8452', edgecolor='black', hatch='...')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Communication Rounds', fontsize=14, fontweight='bold')
ax.set_xlabel('Client Selection Methods', fontsize=14, fontweight='bold')
#ax.set_title('Convergence Speed Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=14, fontweight='bold')
ax.legend(fontsize=14)

# Add a light grid behind the bars
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)

# Function to attach a text label above each bar
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

# Save the figure
plt.savefig('convergence_speed.pdf', dpi=300) # Best for LaTeX
plt.savefig('convergence_speed.png', dpi=300) # For preview
plt.show()