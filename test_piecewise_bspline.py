from time import time
import numpy as np
import matplotlib.pyplot as plt
from trajectorygenerator.piecewise_bsplines import PiecewiseBsplineEvaluation

order = 2
# # 1 dimensions
# control_points_1 = np.array([1,2,3])
# control_points_2 = np.array([13.4,12,4,15,15.6,17.7])
# control_points_3 = np.array([21,22,23.2,24])
# # 2 dimensions
# control_points_1 = np.array([[1,2,3],[1.2,3.3,4.4]])
# control_points_2 = np.array([[13.4,12,4,15,15.6,17.7],[16,17.6,13.4,11,14,13.3]])
# control_points_3 = np.array([[21,22,23.2,24],[23,22,24.5,24.3]])
# 3 dimensions
control_points_1 = np.array([[1,2,3],[1.2,3.3,4.4],[1.1,2.2,3.1]])
control_points_2 = np.array([[13.4,12,4,15,15.6,17.7],[16,17.6,13.4,11,14,13.3],[13.1,11,4,16,16.6,13.7]])
control_points_3 = np.array([[21,22,23.2,24],[23,22,24.5,24.3],[21.1,22.4,23,25]])

control_points_list = [control_points_1, control_points_2, control_points_3]
scale_factor_list = [1,2,1.5]
start_time = 2
piecewise_bspline = PiecewiseBsplineEvaluation(order,control_points_list,scale_factor_list,start_time,False)
spline_data, time_data = piecewise_bspline.get_spline_data(1000)
number_of_data_points = 1000
piecewise_bspline.plot_splines(number_of_data_points)