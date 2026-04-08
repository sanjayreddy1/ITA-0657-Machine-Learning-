import pandas as pd
import math

# Load dataset
data = pd.DataFrame([
    ['Sunny','Hot','High','Weak','No'],
    ['Sunny','Hot','High','Strong','No'],
    ['Overcast','Hot','High','Weak','Yes'],
    ['Rain','Mild','High','Weak','Yes'],
    ['Rain','Cool','Normal','Weak','Yes'],
    ['Rain','Cool','Normal','Strong','No'],
    ['Overcast','Cool','Normal','Strong','Yes'],
    ['Sunny','Mild','High','Weak','No'],
    ['Sunny','Cool','Normal','Weak','Yes'],
    ['Rain','Mild','Normal','Weak','Yes']
], columns=['Outlook','Temp','Humidity','Wind','Play'])

# Entropy function
def entropy(col):
    values = col.value_counts()
    total = len(col)
    return -sum((v/total) * math.log2(v/total) for v in values)

# Information Gain
def info_gain(data, attr, target):
    total_entropy = entropy(data[target])
    vals = data[attr].unique()
    
    weighted_entropy = 0
    for v in vals:
        subset = data[data[attr] == v]
        weighted_entropy += (len(subset)/len(data)) * entropy(subset[target])
    
    return total_entropy - weighted_entropy

# ID3 Algorithm
def id3(data, attributes, target):
    # If all same class
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    # If no attributes left
    if len(attributes) == 0:
        return data[target].mode()[0]

    # Choose best attribute
    gains = [info_gain(data, attr, target) for attr in attributes]
    best_attr = attributes[gains.index(max(gains))]

    tree = {best_attr: {}}

    for val in data[best_attr].unique():
        subset = data[data[best_attr] == val]
        if subset.empty:
            tree[best_attr][val] = data[target].mode()[0]
        else:
            remaining_attrs = [a for a in attributes if a != best_attr]
            tree[best_attr][val] = id3(subset, remaining_attrs, target)

    return tree

# Build tree
attributes = list(data.columns[:-1])
tree = id3(data, attributes, 'Play')

print("Decision Tree:")
print(tree)