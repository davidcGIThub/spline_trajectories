
import numpy as np
import matplotlib.pyplot as plt
from bsplinegenerator.bsplines import BsplineEvaluation
from trajectorygenerator.trajectory_generator import TrajectoryGenerator
from trajectorygenerator.piecewise_bsplines import PiecewiseBsplineEvaluation

order = 3
dimension = 2
derivative_order = 1
# trajectory_gen = ConstrainedTrajectory(ObjectiveType.MINIMIZE_SNAP, dimension)
# trajectory_gen = ConstrainedTrajectory(ObjectiveType.MINIMIZE_TIME_AND_DISTANCE, dimension)
# trajectory_gen = ConstrainedTrajectory(ObjectiveType.MINIMIZE_ACCELERATION, dimension)
trajectory_gen = TrajectoryGenerator(order)
waypoints = np.array([[1,5,9],[2,4,5]])
velocity_waypoints = np.array([[5,0,0],[0,5,5]])
# max_desired_curvature = 
max_velocity = 8
max_acceleration = 10
max_turn_rate = 20
start_time = 0
control_point_list, scale_factor_list = trajectory_gen.generate_trajectory(waypoints,velocity_waypoints,max_velocity,max_acceleration,max_turn_rate)
print("control_point_list : ", control_point_list)
print("scale_factor_list : ", scale_factor_list)
print("max_curvature = " , max_turn_rate/max_velocity)

piecewise_bspline = PiecewiseBsplineEvaluation(order, control_point_list, scale_factor_list,start_time)
num_data_points = 1000
piecewise_bspline.plot_splines(num_data_points)
# piecewise_bspline.plot_spline_vs_time(num_data_points)
piecewise_bspline.plot_derivative(num_data_points,1)
piecewise_bspline.plot_derivative(num_data_points,2)
piecewise_bspline.plot_curvature(num_data_points)
