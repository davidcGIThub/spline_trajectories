
import numpy as np
import matplotlib.pyplot as plt
from bsplinegenerator.bsplines import BsplineEvaluation
from trajectorygenerator.trajectory_generator import TrajectoryGenerator
from trajectorygenerator.piecewise_bsplines import PiecewiseBsplineEvaluation

order = 5
dimension = 2
derivative_order = 1
# trajectory_gen = ConstrainedTrajectory(ObjectiveType.MINIMIZE_SNAP, dimension)
# trajectory_gen = ConstrainedTrajectory(ObjectiveType.MINIMIZE_TIME_AND_DISTANCE, dimension)
# trajectory_gen = ConstrainedTrajectory(ObjectiveType.MINIMIZE_ACCELERATION, dimension)
trajectory_gen = TrajectoryGenerator(order)
waypoints = np.array([[1,5,9],[2,4,5]])
velocity_waypoints = np.array([[0,5,0],[5,0,8]])
max_velocity = 10
max_acceleration = 10
start_time = 0
control_point_list, scale_factor_list = trajectory_gen.generate_trajectory(waypoints,velocity_waypoints,max_velocity,max_acceleration)
print("control_point_list : ", control_point_list)
print("scale_factor_list : ", scale_factor_list)


piecewise_bspline = PiecewiseBsplineEvaluation(order, control_point_list, scale_factor_list,start_time)
num_data_points = 1000
piecewise_bspline.plot_splines(num_data_points)
piecewise_bspline.plot_spline_vs_time(num_data_points)
piecewise_bspline.plot_derivative(num_data_points,1)
piecewise_bspline.plot_derivative(num_data_points,2)

# initial_control = trajectory_gen.get_initial_control_points(waypoints)
# plt.scatter(initial_control[0,:], initial_control[1,:])
# plt.scatter(waypoints[0,:],waypoints[1,:])
# plt.show()