from time import time
import numpy as np
import matplotlib.pyplot as plt
from trajectorygenerator.piecewise_bsplines import PiecewiseBsplineEvaluation

# # 1 dimensions
# control_points_1 = np.array([1,2,3])
# control_points_2 = np.array([13.4,12,4,15,15.6,17.7])
# control_points_3 = np.array([21,22,23.2,24])
# # 2 dimensions
# control_points_1 = np.array([[1,2,3],[1.2,3.3,4.4]])
# control_points_2 = np.array([[13.4,12,4,15,15.6,17.7],[16,17.6,13.4,11,14,13.3]])
# control_points_3 = np.array([[21,22,23.2,24],[23,22,24.5,24.3]])
# # 3 dimensions
# control_points_1 = np.array([[1,2,3],[1.2,3.3,4.4],[1.1,2.2,3.1]])
# control_points_2 = np.array([[13.4,12,4,15,15.6,17.7],[16,17.6,13.4,11,14,13.3],[13.1,11,4,16,16.6,13.7]])
# control_points_3 = np.array([[21,22,23.2,24],[23,22,24.5,24.3],[21.1,22.4,23,25]])
# 2 dimensions clamped
control_points_1 = np.array([[1,2,3,4],[1.2,3.3,4.4,5]])
control_points_2 = np.array([[3,12,4,15,15.6,21],[4.4,17.6,13.4,11,14,23.3]])
control_points_3 = np.array([[21,22,23.2,24],[23.3,22,24.5,24.3]])

control_points_list = [control_points_1, control_points_2, control_points_3]
scale_factor_list = [1,2,1.5]
start_time = 2
derivative_order = 1
order = 3

piecewise_bspline = PiecewiseBsplineEvaluation(order,control_points_list,scale_factor_list,start_time,True)
spline_data, time_data = piecewise_bspline.get_spline_data(1000)
number_of_data_points = 1000
piecewise_bspline.plot_splines(number_of_data_points)
piecewise_bspline.plot_spline_vs_time(number_of_data_points)
piecewise_bspline.plot_derivative(number_of_data_points, derivative_order)