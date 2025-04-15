import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Create the fuzzy variables (inputs and outputs)
current_temp = ctrl.Antecedent(np.arange(0, 51, 1), 'current_temp')  # 0-50°C range
desired_temp = ctrl.Antecedent(np.arange(0, 51, 1), 'desired_temp')  # 0-50°C range
power_adjustment = ctrl.Consequent(np.arange(-100, 101, 1), 'power_adjustment')  # -100% to +100%

# Define membership functions for current temperature (in °C)
current_temp['very_cold'] = fuzz.trimf(current_temp.universe, [0, 0, 10])
current_temp['cold'] = fuzz.trimf(current_temp.universe, [5, 10, 15])
current_temp['cool'] = fuzz.trimf(current_temp.universe, [10, 15, 20])
current_temp['comfortable'] = fuzz.trimf(current_temp.universe, [18, 22, 25])
current_temp['warm'] = fuzz.trimf(current_temp.universe, [22, 27, 30])
current_temp['hot'] = fuzz.trimf(current_temp.universe, [27, 35, 35])
current_temp['very_hot'] = fuzz.trimf(current_temp.universe, [30, 50, 50])

# Define membership functions for desired temperature (in °C)
desired_temp['very_cold'] = fuzz.trimf(desired_temp.universe, [0, 0, 10])
desired_temp['cold'] = fuzz.trimf(desired_temp.universe, [5, 10, 15])
desired_temp['cool'] = fuzz.trimf(desired_temp.universe, [10, 15, 20])
desired_temp['comfortable'] = fuzz.trimf(desired_temp.universe, [18, 22, 25])
desired_temp['warm'] = fuzz.trimf(desired_temp.universe, [22, 27, 30])
desired_temp['hot'] = fuzz.trimf(desired_temp.universe, [27, 35, 35])
desired_temp['very_hot'] = fuzz.trimf(desired_temp.universe, [30, 50, 50])

# Define membership functions for power adjustment
power_adjustment['max_cooling'] = fuzz.trimf(power_adjustment.universe, [-100, -100, -75])
power_adjustment['strong_cooling'] = fuzz.trimf(power_adjustment.universe, [-100, -75, -50])
power_adjustment['moderate_cooling'] = fuzz.trimf(power_adjustment.universe, [-75, -50, -25])
power_adjustment['slight_cooling'] = fuzz.trimf(power_adjustment.universe, [-50, -25, 0])
power_adjustment['no_change'] = fuzz.trimf(power_adjustment.universe, [-25, 0, 25])
power_adjustment['slight_heating'] = fuzz.trimf(power_adjustment.universe, [0, 25, 50])
power_adjustment['moderate_heating'] = fuzz.trimf(power_adjustment.universe, [25, 50, 75])
power_adjustment['strong_heating'] = fuzz.trimf(power_adjustment.universe, [50, 75, 100])
power_adjustment['max_heating'] = fuzz.trimf(power_adjustment.universe, [75, 100, 100])

# Visualize the membership functions (optional)
current_temp.view()
desired_temp.view()
power_adjustment.view()
plt.show()

# Create fuzzy rules
rules = [
    # Very cold current temperature cases
    ctrl.Rule(current_temp['very_cold'] & desired_temp['comfortable'], power_adjustment['strong_heating']),
    ctrl.Rule(current_temp['very_cold'] & desired_temp['warm'], power_adjustment['max_heating']),
    ctrl.Rule(current_temp['very_cold'] & desired_temp['hot'], power_adjustment['max_heating']),
    
    # Cold current temperature cases
    ctrl.Rule(current_temp['cold'] & desired_temp['comfortable'], power_adjustment['moderate_heating']),
    ctrl.Rule(current_temp['cold'] & desired_temp['warm'], power_adjustment['strong_heating']),
    ctrl.Rule(current_temp['cold'] & desired_temp['hot'], power_adjustment['max_heating']),
    
    # Comfortable current temperature cases
    ctrl.Rule(current_temp['comfortable'] & desired_temp['very_cold'], power_adjustment['strong_cooling']),
    ctrl.Rule(current_temp['comfortable'] & desired_temp['cold'], power_adjustment['moderate_cooling']),
    ctrl.Rule(current_temp['comfortable'] & desired_temp['cool'], power_adjustment['slight_cooling']),
    ctrl.Rule(current_temp['comfortable'] & desired_temp['warm'], power_adjustment['slight_heating']),
    ctrl.Rule(current_temp['comfortable'] & desired_temp['hot'], power_adjustment['moderate_heating']),
    
    # Warm current temperature cases
    ctrl.Rule(current_temp['warm'] & desired_temp['very_cold'], power_adjustment['max_cooling']),
    ctrl.Rule(current_temp['warm'] & desired_temp['cold'], power_adjustment['strong_cooling']),
    ctrl.Rule(current_temp['warm'] & desired_temp['cool'], power_adjustment['moderate_cooling']),
    ctrl.Rule(current_temp['warm'] & desired_temp['comfortable'], power_adjustment['slight_cooling']),
    
    # Hot current temperature cases
    ctrl.Rule(current_temp['hot'] & desired_temp['very_cold'], power_adjustment['max_cooling']),
    ctrl.Rule(current_temp['hot'] & desired_temp['cold'], power_adjustment['max_cooling']),
    ctrl.Rule(current_temp['hot'] & desired_temp['cool'], power_adjustment['strong_cooling']),
    ctrl.Rule(current_temp['hot'] & desired_temp['comfortable'], power_adjustment['moderate_cooling'])
]

# Create and simulate the control system
temp_ctrl = ctrl.ControlSystem(rules)
temp_simulation = ctrl.ControlSystemSimulation(temp_ctrl)

# Example usage with Celsius values
temp_simulation.input['current_temp'] = 28  # Current temperature in °C
temp_simulation.input['desired_temp'] = 22  # Desired temperature in °C

# Compute the result
temp_simulation.compute()

# Print the output
print("Power Adjustment: %.3f" %temp_simulation.output['power_adjustment'], '%')
power_adjustment.view(sim=temp_simulation)
plt.show()