import pandas as pd

# Constants (container capacities)
MAX_WEIGHT = 45000  # Maximum weight capacity
MAX_VOLUME = 3600   # Maximum volume capacity
MAX_PALLETS = 60    # Maximum pallet capacity

def compute_metric(data):
    """
    Computes the fitting metric for each order based on its weight, volume, and pallets.

    Args:
        data (pd.DataFrame): A DataFrame containing the orders with 'Weight (lbs)', 'Volume (in3)', and 'Pallets' columns.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'Metric' column.
    """
    # Compute the metric for each order
    data['Metric'] = (data['Weight (lbs)'] / MAX_WEIGHT +
                      data['Volume (in3)'] / MAX_VOLUME +
                      data['Pallets'] / MAX_PALLETS)
    return data

def greedy_initial_solution(data):
    """
    Creates an initial greedy solution for the container optimization problem.

    Args:
        data (pd.DataFrame): The orders DataFrame with 'Metric' computed.

    Returns:
        dict: A dictionary mapping container IDs to the list of assigned orders.
    """
    # Sort orders by metric in descending order
    data = data.sort_values(by='Metric', ascending=False)

    # Containers
    containers = {}
    container_id = 0

    # Tracking remaining capacity for each container
    remaining_capacity = []

    # Assign orders greedily
    for _, order in data.iterrows():
        assigned = False
        for i, (remaining_weight, remaining_volume, remaining_pallets) in enumerate(remaining_capacity):
            if (order['Weight (lbs)'] <= remaining_weight and
                order['Volume (in3)'] <= remaining_volume and
                order['Pallets'] <= remaining_pallets):
                containers[i].append(int(order['Order Number']))
                remaining_capacity[i] = (
                    remaining_weight - order['Weight (lbs)'],
                    remaining_volume - order['Volume (in3)'],
                    remaining_pallets - order['Pallets']
                )
                assigned = True
                break
        
        if not assigned:
            containers[container_id] = [int(order['Order Number'])]
            remaining_capacity.append((
                MAX_WEIGHT - order['Weight (lbs)'],
                MAX_VOLUME - order['Volume (in3)'],
                MAX_PALLETS - order['Pallets']
            ))
            container_id += 1

    print(containers)
    return containers
