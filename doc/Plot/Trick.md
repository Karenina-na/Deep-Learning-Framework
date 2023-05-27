# 🎭 Trick for matplotlib

This repository is a collection of tips and tricks related to using the matplotlib plotting library. Its purpose is to assist users in utilizing matplotlib more efficiently.

## 📝 Instructions for use

You can employ the calling methods provided in this README to embellish your scientific plots.

## 📃 List of techniques

### Adjust the offset between the x-axis and y-axis titles and their respective axes.

```python
# set y-axis
plt.gca().yaxis.set_label_coords(offset_for_y, 0.5)

plt.gca().xaxis.set_label_coords(0.5, offset_for_x)
```

### 