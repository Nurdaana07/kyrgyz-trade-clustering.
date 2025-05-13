import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Данные
data = {
    'Country': ['China', 'Russia', 'Kazakhstan', 'UK', 'Uzbekistan'],
    'Export': [123.6, 1000, 418.4, 1200, 346.9],
    'Import': [5400, 2500, 921.6, 55.5, 505.9],
    'Total_Trade': [5523.6, 3500, 1340, 1255.5, 852.8]
}

df = pd.DataFrame(data)
X = df[['Export', 'Import', 'Total_Trade']]

# Нормализация
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Кластеризация
linked = linkage(X_scaled, method='ward')

# Дендрограмма
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=df['Country'].values)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Country')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()
