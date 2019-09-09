#!/usr/bin/env python
# coding: utf-8

# In[54]:


#import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score


# In[55]:


df = pd.read_csv("Treasury.csv")
df


# In[56]:


df.drop(df.columns[0:2], axis=1)
df_x = df.iloc[:,5:7]
df_y = df.iloc[:,-1]


# In[57]:


#Split data with test size 30% of total observations
from sklearn.model_selection import train_test_split
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(df_x, df_y, test_size=0.3,
random_state=1, stratify=df_y)


# In[58]:


#Standardize units
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train_knn)
X_train_sc = sc.transform(X_train_knn)
X_test_sc = sc.transform(X_test_knn)


# In[59]:


from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx = None,
resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')


# In[60]:


from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,26)
score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(X_train_sc, y_train_knn)
    y_pred = knn.predict(X_test_sc)
    score.append(accuracy_score(y_test_knn, y_pred))


# In[61]:


optimal_k = score.index(np.max(score)) + 1


# In[62]:


optimal_k


# In[63]:


score


# In[64]:


knn_nk = KNeighborsClassifier(n_neighbors=optimal_k)
knn_nk.fit(X_train_sc, y_train_knn)
X_combined_std = np.vstack((X_train_sc, X_test_sc))
y_combined = np.hstack((y_train_knn, y_test_knn))
plot_decision_regions(X_combined_std, y_combined, classifier=knn_nk, test_idx=range(105,150))
plt.xlabel('ctd_last_first')
plt.ylabel('ctd1_percent')
plt.legend(loc='upper left')
plt.show()


# In[65]:


#Establish decision tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
df_x_dt = df.iloc[:,2:4]
df_y_dt = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(df_x_dt, df_y_dt, test_size=0.3, random_state=1, stratify=df_y_dt)
sc = StandardScaler()
sc.fit(X_train)
X_train_sc_tree = sc.transform(X_train)
X_test_sc_tree = sc.transform(X_test)
tree = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state
=1)
tree.fit(X_train_sc_tree, y_train)
X_combined = np.vstack((X_train_sc_tree, X_test_sc_tree))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined ,y_combined, classifier=tree, test_idx=
range(105, 150))
plt.xlabel('roll_start')
plt.ylabel('roll_heart')
plt.legend(loc='upper left')
plt.show()


# In[53]:


print("My name is Zihan Yu")
print("My NetID is: zihanyu3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




