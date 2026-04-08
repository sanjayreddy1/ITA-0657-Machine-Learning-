def find_s_algorithm(training_data):
    hypothesis = ['0'] * (len(training_data[0]) - 1)

    print("Initial Hypothesis:", hypothesis)
    print("-" * 50)

    for i, sample in enumerate(training_data):
        if sample[-1] == "Yes":  
            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':
                    hypothesis[j] = sample[j]
                elif hypothesis[j] != sample[j]:
                    hypothesis[j] = '?'
        
        print(f"Step {i+1} Hypothesis:", hypothesis)

    return hypothesis


training_data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]


final_hypothesis = find_s_algorithm(training_data)

print("\nFinal Hypothesis:", final_hypothesis)
