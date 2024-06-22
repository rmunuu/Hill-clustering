import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

with open('data_num.pkl', 'rb') as f:
    data_num = pkl.load(f)

# This will store the data points
data = []

def onclick(event):
    # Only add points when the click is within the axes
    if event.inaxes:
        x, y = event.xdata, event.ydata
        data.append((x, y))
        ax.plot(x, y, 'ro')  # 'ro' for red dots
        fig.canvas.draw()

# Create a figure and a set of subplots
fig, ax = plt.subplots()
ax.set_xlim(0, 1)  # Set x-axis limits
ax.set_ylim(0, 1)  # Set y-axis limits
ax.set_title('Click to add points')

# Connect the click event to the onclick function
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

data = np.array(data)


# Save the data points to a pickle file
with open(f'./data_pickle/data_{data_num}.pkl', 'wb') as f:
    pkl.dump(data, f)

data_num += 1
with open('data_num.pkl', 'wb') as f:
    pkl.dump(data_num, f)

print(f"data_num: {data_num}")

plt.scatter(data[:, 0], data[:, 1])
plt.savefig(f'./data_plot/data_{data_num}.png')