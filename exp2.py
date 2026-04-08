import csv
import copy

# Load CSV data
def load_data(filename):
    with open(filename, 'r') as f:
        data = list(csv.reader(f))
    return data

# Initialize S and G
def initialize_hypotheses(num_attributes):
    S = ['0'] * num_attributes
    G = [['?'] * num_attributes]
    return S, G

# Check if hypothesis is consistent with example
def is_consistent(h, x):
    return all(h[i] == '?' or h[i] == x[i] for i in range(len(h)))

# Generalize S
def generalize_S(S, x):
    for i in range(len(S)):
        if S[i] == '0':
            S[i] = x[i]
        elif S[i] != x[i]:
            S[i] = '?'
    return S

# Specialize G
def specialize_G(G, S, x):
    new_G = []
    for g in G:
        if is_consistent(g, x):
            for i in range(len(g)):
                if g[i] == '?':
                    if S[i] != x[i]:
                        new_h = g.copy()
                        new_h[i] = S[i]
                        new_G.append(new_h)
    return new_G

# Remove overly general hypotheses
def remove_more_general(G):
    final_G = []
    for g in G:
        if not any(all(g[i] == '?' or g[i] == other[i] for i in range(len(g))) and g != other for other in G):
            final_G.append(g)
    return final_G

# Candidate Elimination Algorithm
def candidate_elimination(data):
    num_attributes = len(data[0]) - 1
    S, G = initialize_hypotheses(num_attributes)

    print("Initial S:", S)
    print("Initial G:", G)
    print("-" * 50)

    for idx, row in enumerate(data):
        x = row[:-1]
        label = row[-1]

        if label == "Yes":  # Positive Example
            G = [g for g in G if is_consistent(g, x)]
            S = generalize_S(S, x)

        else:  # Negative Example
            G = specialize_G(G, S, x)

        G = remove_more_general(G)

        print(f"Step {idx+1}")
        print("S:", S)
        print("G:", G)
        print("-" * 50)

    return S, G


# Run
data = load_data(r"C:\Users\sanja\OneDrive\Desktop\ML PROGRAMS\data.csv")
S_final, G_final = candidate_elimination(data[1:])  # skip header

print("\nFinal Specific Boundary S:", S_final)
print("Final General Boundary G:", G_final)