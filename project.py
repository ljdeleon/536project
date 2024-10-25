import pandas as pd
import pulp

# Load the dataset
data = pd.read_csv('test data.csv')

# Define constants
MAX_WEIGHT = 45000
MAX_VOLUME = 3600
MAX_PALLETS = 60
MAX_CONTAINERS = len(data)  # Maximum possible containers in worst case

# Initialize the optimization problem
prob = pulp.LpProblem("Container_Optimization", pulp.LpMinimize)

# Extract data from DataFrame
weights = data['Weight (lbs)'].tolist()
volumes = data['Volume (in3)'].tolist()
pallets = data['Pallets'].tolist()
order_numbers = data['Order Number'].tolist()

# Decision variables
x = pulp.LpVariable.dicts("x", [(i, j) for i in range(len(data)) for j in range(MAX_CONTAINERS)], cat="Binary")
y = pulp.LpVariable.dicts("y", [j for j in range(MAX_CONTAINERS)], cat="Binary")

# Objective: minimize the number of containers used
prob += pulp.lpSum([y[j] for j in range(MAX_CONTAINERS)])

# Constraint 1: Each order is assigned to exactly one container
for i in range(len(data)):
    prob += pulp.lpSum([x[i, j] for j in range(MAX_CONTAINERS)]) == 1

# Constraint 2: Weight capacity of each container
for j in range(MAX_CONTAINERS):
    prob += pulp.lpSum([weights[i] * x[i, j] for i in range(len(data))]) <= MAX_WEIGHT * y[j]

# Constraint 3: Volume capacity of each container
for j in range(MAX_CONTAINERS):
    prob += pulp.lpSum([volumes[i] * x[i, j] for i in range(len(data))]) <= MAX_VOLUME * y[j]

# Constraint 4: Pallet capacity of each container
for j in range(MAX_CONTAINERS):
    prob += pulp.lpSum([pallets[i] * x[i, j] for i in range(len(data))]) <= MAX_PALLETS * y[j]

# Solve the problem
prob.solve()

# Output results
print(f"Status: {pulp.LpStatus[prob.status]}")

# Containers used
containers_used = sum([y[j].varValue for j in range(MAX_CONTAINERS)])
print(f"Total Containers Used: {containers_used}")

# Orders in each container
for j in range(MAX_CONTAINERS):
    if y[j].varValue == 1:
        print(f"Container {j+1}:")
        for i in range(len(data)):
            if x[i, j].varValue == 1:
                print(f"  Order {order_numbers[i]} - Weight: {weights[i]}, Volume: {volumes[i]}, Pallets: {pallets[i]}")
