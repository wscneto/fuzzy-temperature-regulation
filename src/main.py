import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Fuzzy variables
current_temp = ctrl.Antecedent(np.arange(0, 51, 1), 'current_temp')  # 0-50°C
water_flow = ctrl.Antecedent(np.arange(0, 101, 1), 'water_flow')     # 0-100%
valve_opening = ctrl.Consequent(np.arange(-100, 101, 1), 'valve_opening')  # -100% to +100%

# Membership functions for current temperature
current_temp['very_cold'] = fuzz.trimf(current_temp.universe, [0, 0, 10])
current_temp['cold'] = fuzz.trimf(current_temp.universe, [5, 10, 15])
current_temp['cool'] = fuzz.trimf(current_temp.universe, [10, 15, 20])
current_temp['comfortable'] = fuzz.trimf(current_temp.universe, [18, 22, 25])
current_temp['warm'] = fuzz.trimf(current_temp.universe, [22, 27, 30])
current_temp['hot'] = fuzz.trimf(current_temp.universe, [27, 35, 40])
current_temp['very_hot'] = fuzz.trimf(current_temp.universe, [35, 50, 50])

# Membership functions for water flow
water_flow['very_weak'] = fuzz.trimf(water_flow.universe, [0, 0, 20])
water_flow['weak'] = fuzz.trimf(water_flow.universe, [10, 25, 40])
water_flow['normal'] = fuzz.trimf(water_flow.universe, [30, 50, 70])
water_flow['strong'] = fuzz.trimf(water_flow.universe, [60, 75, 85])
water_flow['very_strong'] = fuzz.trimf(water_flow.universe, [80, 100, 100])

# Membership functions for valve opening
valve_opening['max_closing'] = fuzz.trimf(valve_opening.universe, [-100, -100, -75])
valve_opening['strong_closing'] = fuzz.trimf(valve_opening.universe, [-100, -75, -50])
valve_opening['moderate_closing'] = fuzz.trimf(valve_opening.universe, [-75, -50, -25])
valve_opening['slight_closing'] = fuzz.trimf(valve_opening.universe, [-50, -25, 0])
valve_opening['no_change'] = fuzz.trimf(valve_opening.universe, [-25, 0, 25])
valve_opening['slight_opening'] = fuzz.trimf(valve_opening.universe, [0, 25, 50])
valve_opening['moderate_opening'] = fuzz.trimf(valve_opening.universe, [25, 50, 75])
valve_opening['strong_opening'] = fuzz.trimf(valve_opening.universe, [50, 75, 100])
valve_opening['max_opening'] = fuzz.trimf(valve_opening.universe, [75, 100, 100])

rules = [
    # Very cold
    ctrl.Rule(current_temp['very_cold'] & water_flow['very_weak'], valve_opening['slight_closing']),
    ctrl.Rule(current_temp['very_cold'] & water_flow['weak'], valve_opening['moderate_closing']),
    ctrl.Rule(current_temp['very_cold'] & water_flow['normal'], valve_opening['strong_closing']),
    ctrl.Rule(current_temp['very_cold'] & water_flow['strong'], valve_opening['max_closing']),
    ctrl.Rule(current_temp['very_cold'] & water_flow['very_strong'], valve_opening['max_closing']),

    # Cold
    ctrl.Rule(current_temp['cold'] & water_flow['very_weak'], valve_opening['slight_closing']),
    ctrl.Rule(current_temp['cold'] & water_flow['weak'], valve_opening['moderate_closing']),
    ctrl.Rule(current_temp['cold'] & water_flow['normal'], valve_opening['moderate_closing']),
    ctrl.Rule(current_temp['cold'] & water_flow['strong'], valve_opening['strong_closing']),
    ctrl.Rule(current_temp['cold'] & water_flow['very_strong'], valve_opening['max_closing']),

    # Cool
    ctrl.Rule(current_temp['cool'] & water_flow['very_weak'], valve_opening['slight_closing']),
    ctrl.Rule(current_temp['cool'] & water_flow['weak'], valve_opening['moderate_closing']),
    ctrl.Rule(current_temp['cool'] & water_flow['normal'], valve_opening['moderate_closing']),
    ctrl.Rule(current_temp['cool'] & water_flow['strong'], valve_opening['slight_closing']),
    ctrl.Rule(current_temp['cool'] & water_flow['very_strong'], valve_opening['no_change']),

    # Comfortable
    ctrl.Rule(current_temp['comfortable'] & water_flow['very_weak'], valve_opening['slight_closing']),
    ctrl.Rule(current_temp['comfortable'] & water_flow['weak'], valve_opening['no_change']),
    ctrl.Rule(current_temp['comfortable'] & water_flow['normal'], valve_opening['no_change']),
    ctrl.Rule(current_temp['comfortable'] & water_flow['strong'], valve_opening['no_change']),
    ctrl.Rule(current_temp['comfortable'] & water_flow['very_strong'], valve_opening['slight_opening']),

    # Warm
    ctrl.Rule(current_temp['warm'] & water_flow['very_weak'], valve_opening['moderate_opening']),
    ctrl.Rule(current_temp['warm'] & water_flow['weak'], valve_opening['slight_opening']),
    ctrl.Rule(current_temp['warm'] & water_flow['normal'], valve_opening['no_change']),
    ctrl.Rule(current_temp['warm'] & water_flow['strong'], valve_opening['slight_closing']),
    ctrl.Rule(current_temp['warm'] & water_flow['very_strong'], valve_opening['moderate_closing']),

    # Hot
    ctrl.Rule(current_temp['hot'] & water_flow['very_weak'], valve_opening['strong_opening']),
    ctrl.Rule(current_temp['hot'] & water_flow['weak'], valve_opening['moderate_opening']),
    ctrl.Rule(current_temp['hot'] & water_flow['normal'], valve_opening['slight_opening']),
    ctrl.Rule(current_temp['hot'] & water_flow['strong'], valve_opening['no_change']),
    ctrl.Rule(current_temp['hot'] & water_flow['very_strong'], valve_opening['slight_closing']),

    # Very hot
    ctrl.Rule(current_temp['very_hot'] & water_flow['very_weak'], valve_opening['max_opening']),
    ctrl.Rule(current_temp['very_hot'] & water_flow['weak'], valve_opening['strong_opening']),
    ctrl.Rule(current_temp['very_hot'] & water_flow['normal'], valve_opening['moderate_opening']),
    ctrl.Rule(current_temp['very_hot'] & water_flow['strong'], valve_opening['slight_opening']),
    ctrl.Rule(current_temp['very_hot'] & water_flow['very_strong'], valve_opening['no_change']),
]

temp_ctrl = ctrl.ControlSystem(rules)
temp_simulation = ctrl.ControlSystemSimulation(temp_ctrl)

# Input
input_temp = 40
input_flow = 32
temp_simulation.input['current_temp'] = input_temp
temp_simulation.input['water_flow'] = input_flow

# Visualize the membership functions
current_temp.view()
plt.axvline(x=input_temp, color='red', linestyle='--', label=f'Input: {input_temp}°C')
plt.legend()
water_flow.view()
plt.axvline(x=input_flow, color='blue', linestyle='--', label=f'Input: {input_flow}%')
plt.legend()
valve_opening.view()
plt.show()

temp_simulation.compute()

print("Valve Opening Adjustment: %.2f%%" % temp_simulation.output['valve_opening'])
valve_opening.view(sim=temp_simulation)
plt.show()

# 3d visualization
temp_labels = ['very_cold', 'cold', 'cool', 'comfortable', 'warm', 'hot', 'very_hot']
flow_labels = ['very_weak', 'weak', 'normal', 'strong', 'very_strong']
temp_values = np.arange(len(temp_labels))
flow_values = np.arange(len(flow_labels))

# negative = closing, 0 = no change, positive = opening
valve_matrix = np.array([
    [-1, -2, -3, -4, -4],   # Very Cold
    [-1, -2, -2, -3, -4],   # Cold
    [-1, -2, -2, -1,  0],   # Cool
    [-1,  0,  0,  0,  1],   # Comfortable
    [ 2,  1,  0, -1, -2],   # Warm
    [ 3,  2,  1,  0, -1],   # Hot
    [ 4,  3,  2,  1,  0]    # Very Hot
])

flow_grid, temp_grid = np.meshgrid(flow_values, temp_values)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(flow_grid, temp_grid, valve_matrix, cmap='coolwarm', edgecolor='k')

ax.set_xticks(flow_values)
ax.set_xticklabels(flow_labels, rotation=45, ha='right')
ax.set_yticks(temp_values)
ax.set_yticklabels(temp_labels)
ax.set_zlabel('Valve Opening Action')
ax.set_xlabel('Water Flow')
ax.set_ylabel('Temperature')
ax.set_title('3d Fuzzy Control Surface')

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)
plt.show()