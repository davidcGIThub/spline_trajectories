import numpy as np
import random
import matplotlib.pyplot as plt
from bsplinegenerator.bsplines import BsplineEvaluation
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from bsplinegenerator.matrix_evaluation import get_M_matrix, get_T_vector, get_T_derivative_vector

### Parameters
start_time = 0
scale_factor = 1
derivative_order = 1
clamped = False
number_data_points = 1000
order = 3
objective_variable_bounds = Bounds(lb=0, ub = 1.0)
minimize_options = {'disp': True}#, 'maxiter': self.maxiter, 'ftol': tol}
t0_1 = np.array([0.0])
t0_2 = np.array([0.5])
t0_3 = np.array([1.])
control_points = np.random.randint(10, size=(random.randint(2, 3),order+1)) # random

# control_points = np.array([[7, 9, 2, 3],[6, 2, 0, 9],[4, 4, 1, 9]])

if order == 2:
    angle = np.pi/2 # 2rd degree
elif order == 3:
    angle = np.pi/2 # 3rd degree
elif order == 4:
    # angle = np.pi/3 # 4th degree
    angle = np.pi/2 # 4th degree
elif order == 5:
    # angle = np.pi/4 # 4th degree
    angle = np.pi/2 # 4th degree
else:
    angle = np.pi/2

def create_random_control_points_greater_than_angles(num_control_points,angle):
    control_points = np.zeros((2,num_control_points))
    len_ave = 3
    for i in range(num_control_points):
        if i == 0:
            control_points[:,i][:,None] = np.array([[0],[0]])
        elif i == 1:
            random_vec = np.random.rand(2,1)
            next_vec = len_ave*random_vec/np.linalg.norm(random_vec)
            control_points[:,i][:,None] = control_points[:,i-1][:,None] + next_vec
        else:
            # new_angle = angle*2*(0.5-np.random.rand())
            new_angle = angle*np.sign((0.5-np.random.rand()))
            # new_angle = angle
            R = np.array([[np.cos(new_angle), -np.sin(new_angle)],[np.sin(new_angle), np.cos(new_angle)]])
            prev_vec = control_points[:,i-1][:,None] - control_points[:,i-2][:,None]
            unit_prev_vec = prev_vec/np.linalg.norm(prev_vec)
            next_vec = len_ave*np.dot(R,unit_prev_vec)*np.random.rand()
            control_points[:,i][:,None] = control_points[:,i-1][:,None] + next_vec
    return control_points

# control_points = create_random_control_points_greater_than_angles(order+1,angle)


### Create B-Spline Object ###
bspline = BsplineEvaluation(control_points, order, start_time, scale_factor, clamped)

####  Evaluate B-Spline Data ###
spline_curvature_data, time_data = bspline.get_spline_curvature_data(number_data_points)
print("control points: " , control_points)
print("max_curvature: " , np.max(spline_curvature_data))


### evaluate max curvature

M = get_M_matrix(0, order, np.array([]), False)

def objective_function(time):
    t = time[0]
    dT = get_T_derivative_vector(order,t,0,1,scale_factor)
    d2T = get_T_derivative_vector(order,t,0,2,scale_factor)
    vel_vec = np.dot(control_points,np.dot(M,dT)).flatten()
    accel_vec = np.dot(control_points,np.dot(M,d2T)).flatten()
    curvature = np.linalg.norm(np.cross(vel_vec,accel_vec))/ np.linalg.norm(vel_vec)**3
    return -curvature

def objective_function_2(time):
    t = time[0]
    dT = get_T_derivative_vector(order,t,0,1,scale_factor)
    vel_vec = np.dot(control_points,np.dot(M,dT))
    velocity_magnitude = np.linalg.norm(vel_vec)
    return velocity_magnitude


result_1 = minimize(
    objective_function_2,
    x0=t0_1,
    method='SLSQP',
    bounds=objective_variable_bounds, 
    options = minimize_options)

result_2 = minimize(
    objective_function_2,
    x0=t0_2,
    method='SLSQP',
    bounds=objective_variable_bounds, 
    options = minimize_options)

result_3 = minimize(
    objective_function_2,
    x0=t0_3,
    method='SLSQP',
    bounds=objective_variable_bounds, 
    options = minimize_options)

print("result_1: " , result_1.x)
print("result_3: " , result_3.x)

velocity_1 = bspline.get_derivative_magnitude_at_time_t(result_1.x,1)
velocity_2 = bspline.get_derivative_magnitude_at_time_t(result_2.x,1)
velocity_3 = bspline.get_derivative_magnitude_at_time_t(result_3.x,1)
min_velocity = min((velocity_1,velocity_2,velocity_3))
if velocity_1 < velocity_3:
    if velocity_1 < velocity_2:
        t_min = result_1.x
    else:
        t_min = result_2.x
elif velocity_2 < velocity_3:
    t_min = result_2.x
else:
    t_min = result_3.x
accel_at_min_velocity = bspline.get_derivative_magnitude_at_time_t(t_min,2)
curvature_1 = bspline.get_curvature_at_time_t(result_1.x)
curvature_2 = bspline.get_curvature_at_time_t(result_2.x)
curvature_3 = bspline.get_curvature_at_time_t(result_3.x)
max_curvature = np.max((curvature_1,curvature_2,curvature_3))
acceleration_1 = bspline.get_derivative_magnitude_at_time_t(result_1.x,2)
acceleration_2 = bspline.get_derivative_magnitude_at_time_t(result_2.x,2)
acceleration_3 = bspline.get_derivative_magnitude_at_time_t(result_3.x,2)
max_acceleration = max((acceleration_1,acceleration_2, acceleration_3))
curvature_data, time_data = bspline.get_spline_curvature_data(number_data_points)
# Plot Spline Data
# bspline.plot_spline(number_data_points)
plt.plot(time_data, curvature_data, label="curvature")
plt.plot(time_data, curvature_data*0 + max_curvature , label="max curvature")
plt.plot(time_data, curvature_data*0 + accel_at_min_velocity/min_velocity**2 , label="psuedo max curvature")
plt.plot(time_data, curvature_data*0 + max_acceleration/min_velocity**2 , label="conservative max curvature")
plt.legend()
plt.show()

bspline.plot_derivative_magnitude(number_data_points,1)

