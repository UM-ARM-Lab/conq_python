import matplotlib.cm as cm
import numpy as np


def viz_predictions(rgb_np, predictions, fig, ax):
    ax.imshow(rgb_np)
    rng = np.random.RandomState(0)
    class_colors = {}
    for pred in predictions:
        class_name = pred["class"]
        if class_name not in class_colors:
            class_colors[class_name] = cm.hsv(rng.uniform())
        poly = pred_to_poly(pred)
        c = class_colors[class_name]
        ax.plot(poly[:, 0], poly[:, 1], c=c, linewidth=2, zorder=1, label=class_name)
    ax.legend()


def pred_to_poly(pred):
    points = pred["points"]
    points = np.array([(p['x'], p['y']) for p in points], dtype=int)
    return points
