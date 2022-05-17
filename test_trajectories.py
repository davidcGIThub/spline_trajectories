
import numpy as np
import matplotlib.pyplot as plt
from bsplinegenerator.bsplines import BsplineEvaluation
from trajectorygenerator.constrained_trajectory import ConstrainedTrajectory, ObjectiveType

order = 5
dimension = 2
trajectory_gen = ConstrainedTrajectory(ObjectiveType.MINIMIZE_TIME_AND_DISTANCE, order, dimension)
waypoints = np.array([[1,4,9],[2,4,5]])
start_time = 0
control_points, scale_factor = trajectory_gen.generate_trajectory(waypoints)
bspline = BsplineEvaluation(control_points, order, start_time, scale_factor, False)
number_data_points = 1000
spline_data, time_data = bspline.get_spline_data(number_data_points)

print("waypoints: " , waypoints)
print("scale_factor: " , scale_factor)

plt.figure("Optimized B-Spline")
plt.scatter(control_points[0,:], control_points[1,:],facecolors='none',edgecolors='tab:green',linewidths=2,label="control points")
plt.plot(control_points[0,:], control_points[1,:],color='tab:green')
plt.plot(spline_data[0,:], spline_data[1,:],label="B-spline")
plt.scatter(waypoints[0,:], waypoints[1,:],label="waypoints")
plt.xlabel('x')
plt.ylabel('y')
plt.title("Optimized 5th order B-Spline")
plt.legend()
plt.show()