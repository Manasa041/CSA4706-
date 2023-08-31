import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
true_labels = np.random.choice([0, 1], size=100, p=[0.4, 0.6])
predicted_labels = np.random.choice([0, 1], size=100, p=[0.6, 0.4])
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(6, 4))
sns.set(font_scale=1.2)  
sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
cmap="hots"
plt.show()
