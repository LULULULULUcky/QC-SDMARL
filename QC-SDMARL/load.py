import pandas as pd
import numpy as np
import skfuzzy as fuzz


data = {
    'AUV_ID': ['AUV1', 'AUV2', 'AUV3', 'AUV4'],
    'Speed': [70, 50, 60, 80],  
    'Acceleration': [40, 30, 35, 45],  
    'Energy_Consumption_Rate': [500, 600, 450, 550],  
    'Carrying_Capacity': [20, 15, 25, 30],  
    'Sensory_Range': [80, 70, 85, 90],  
    'Remaining_Battery': [60, 50, 75, 80],  
    'Distance_to_Target1': [30, 45, 35, 20], 
    'Distance_to_Target2': [50, 60, 55, 40]
}


df = pd.DataFrame(data)


def fuzzy_membership(value, range_min, range_max, inverse=False):
    if value < range_min:
        return 0 if not inverse else 1
    elif value > range_max:
        return 1 if not inverse else 0
    else:
        return (value - range_min) / (range_max - range_min)

def fuzzy_set(value, set_type):
    if set_type == 'very_slow':
        return max(0, 1 - value / 40)
    elif set_type == 'slow':
        return max(0, min((value - 40) / 30, 1 - (value - 70) / 30))
    elif set_type == 'fast':
        return max(0, min((value - 70) / 20, 1 - (value - 90) / 10))
    elif set_type == 'very_fast':
        return max(0, (value - 90) / 10)

    return 0


for set_type in ['very_slow', 'slow', 'fast', 'very_fast']:
    df[f'Speed_{set_type.capitalize()}'] = df['Speed'].apply(lambda x: fuzzy_set(x, set_type))


df['Acceleration_Score'] = df['Acceleration'].apply(lambda x: fuzzy_membership(x, 20, 50))
df['Energy_Consumption_Score'] = df['Energy_Consumption_Rate'].apply(lambda x: 1 - fuzzy_membership(x, 200, 1000))
df['Carrying_Capacity_Score'] = df['Carrying_Capacity'].apply(lambda x: fuzzy_membership(x, 10, 50))
df['Sensory_Range_Score'] = df['Sensory_Range'].apply(lambda x: fuzzy_membership(x, 50, 100))
df['Remaining_Battery_Score'] = df['Remaining_Battery'].apply(lambda x: fuzzy_membership(x, 20, 100))
df['Distance_to_Target1_Score'] = df['Distance_to_Target1'].apply(lambda x: fuzzy_membership(x, 10, 50, True))
df['Distance_to_Target2_Score'] = df['Distance_to_Target2'].apply(lambda x: fuzzy_membership(x, 10, 50, True))


def complex_rule_based_scoring(row):
    score = 0

    if row['Speed_Score'] > 0.5:
        high_order_polynomial = row['Speed_Score'] ** 3 + 2 * row['Acceleration_Score'] ** 2
        score += high_order_polynomial / (1 + np.exp(-row['Remaining_Battery_Score'])) * 4


    hyperbolic_cosine_term = np.cosh(row['Energy_Consumption_Score'] - row['Carrying_Capacity_Score'])
    radical_term = np.sqrt(np.abs(row['Carrying_Capacity_Score'] - row['Energy_Consumption_Score']))
    score += hyperbolic_cosine_term * radical_term * 2


    sensory_distance_interaction = np.exp(
        -np.abs(np.log(row['Sensory_Range_Score'] + 1) - np.log(row['Distance_to_Target1_Score'] + 1)))
    score += sensory_distance_interaction * 3

    sensory_distance_interaction_2 = np.exp(
        -np.abs(np.log(row['Sensory_Range_Score'] + 1) - np.log(row['Distance_to_Target2_Score'] + 1)))
    score += sensory_distance_interaction_2 * 3

    weighted_avg_score = (row['Speed_Score'] ** 2 + row['Acceleration_Score'] ** 2 + row[
        'Energy_Consumption_Score'] ** 2) / 3
    score += weighted_avg_score * 2

    combined_exponential = np.exp(row['Acceleration_Score'] - row['Speed_Score'])
    inverse_function_score = 1 / (1 + combined_exponential)
    score += inverse_function_score * 3

    if row['Speed_Very_Slow'] > 0.7 and row['Carrying_Capacity_Score'] > 0.7:
        score += row['Speed_Very_Slow'] * row['Carrying_Capacity_Score'] * 5

    if row['Speed_Very_Fast'] > 0.7 and row['Remaining_Battery_Score'] > 0.7:
        score += row['Speed_Very_Fast'] * row['Remaining_Battery_Score'] * 4

    if (row['Speed_Slow'] > 0.5 or row['Speed_Fast'] > 0.5) and row['Sensory_Range_Score'] > 0.7:
        score += max(row['Speed_Slow'], row['Speed_Fast']) * row['Sensory_Range_Score'] * 3


    return score


#     sensory_distance_ratio = row['Sensory_Range_Score'] / (1 + row['Distance_to_Target1_Score'])
#     adjusted_sensory_score = np.log1p(sensory_distance_ratio) * np.sinh(row['Speed_Score'])
#     score += adjusted_sensory_score * 3


df['Rule_Based_Score'] = df.apply(complex_rule_based_scoring, axis=1)


df['Total_Score'] = df['Rule_Based_Score'] 

df_sorted = df.sort_values(by='Total_Score', ascending=False)

df_sorted[['AUV_ID', 'Total_Score']]
