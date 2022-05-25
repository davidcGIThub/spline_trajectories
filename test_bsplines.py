import numpy as np
import random
import matplotlib.pyplot as plt
from bsplinegenerator.bsplines import BsplineEvaluation

### Control Points ###
# control_points = np.array([1,2,2.5,4,5.2,6,6.3,5]) # 1 D
control_points = np.array([[-3,-4,-2,-.5,1,0,2,3.5,3],[.5,3.5,6,5.5,3.7,2,-1,2,5]]) # 2 D
leg1 = 1
leg2 = .001
outer_legs = 1
# control_points = np.array([[0,leg1,leg1],[0,0,leg2]]) # 2nd order
# control_points = np.array([[0,leg1,leg1+leg2*np.cos(np.pi/4)],[0,0,leg2*np.sin(np.pi/4)]]) # 2nd order
# control_points = np.array([[-0.1,0,leg1,leg1],[0,0,0,leg2]]) # 3rd order
# control_points = np.array([[-2*leg1,-leg1,0,leg1,leg1],[0,0,0,0,leg2]]) # 4th order
# control_points = np.array([[-3*outer_legs,-2*outer_legs,-outer_legs,0,leg1,leg1],[0,0,0,0,0,leg2]]) # 5th order
# control_points = np.array([[-3*leg1, -2*leg1, -leg1, 0, 0, 0],[leg2,0,0,0,leg2,leg2]]) # 5th order
# control_points = np.array([[0,0,5,5],[0,5,5,2.5]]) # 2 D
# control_points = np.array([[0,0,5,5,0,],[0,5,5,0,0]]) # 2 D
# control_points = np.array([[-3,-4,-2,-.5,1,0,2,3.5,3],[.5,3.5,6,5.5,3.7,2,-1,2,5],[1,3.2,5,0,3.3,1.5,-1,2.5,4]]) # 3 D
# control_points = np.random.randint(10, size=(random.randint(1, 3),13)) # random
 
if len(control_points) == 1:
    control_points = control_points.flatten()

### Parameters
order = 5
start_time = 0
scale_factor = 0.00100000035366
derivative_order = 1
clamped = False
number_data_points = 1000

### Create B-Spline Object ###
bspline = BsplineEvaluation(control_points, order, start_time, scale_factor, clamped)

####  Evaluate B-Spline Data ###
spline_data, time_data = bspline.get_spline_data(number_data_points)
spline_derivative_data, time_data = bspline.get_spline_derivative_data(number_data_points,derivative_order)
spline_curvature_data, time_data = bspline.get_spline_curvature_data(number_data_points)
basis_function_data, time_data = bspline.get_basis_function_data(number_data_points)
knot_points = bspline.get_knot_points()
defined_knot_points = bspline.get_defined_knot_points()
spline_at_knot_points = bspline.get_spline_at_knot_points()
print("control points: " , control_points)
print("knot_points: " , knot_points)
print("defined knot points: " , defined_knot_points)
print("spline at knots: " , spline_at_knot_points)
print("max_derivative: " , np.max(spline_derivative_data))
print("max_curvature: " , np.max(spline_curvature_data))
theta = np.arctan2(2*leg1, leg2)
print("theta: " , theta)
print("min_diameter: " , 1/np.max(spline_curvature_data)*2)
print("correlation: " , 1/np.max(spline_curvature_data)*2/np.sin(theta))
print("number_of_basis_functions: " , len(basis_function_data))

# Plot Spline Data
bspline.plot_spline(number_data_points)
bspline.plot_spline_vs_time(number_data_points)
bspline.plot_basis_functions(number_data_points)
bspline.plot_derivative(number_data_points, derivative_order)
bspline.plot_curvature(number_data_points)


