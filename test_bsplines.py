import numpy as np
import time
import matplotlib.pyplot as plt
from helper_functions import count_number_of_control_points, get_dimension
from bsplines import BsplineEvaluation

# control_points = np.array([[0,3,-2,-.5,1,0,2,3,3,3],[0,4,6,5.5,3.7,2,-1,5,5,5]]) # initial velocity of 25
# control_points = np.random.randint(10, size=(2,13)) # random
# control_points = np.array([[-3,-3,-3,-2,-.5,1,0,2,3,3,3],[.5,.5,.5,6,5.5,3.7,2,-1,5,5,5]]) # zero initial velocity
# control_points = np.array([[-.5,1,0],[5.5,3.7,2]]) # 3 control points
# control_points = np.array([[-2,-.5,1,0],[6,5.5,3.7,2]]) # 4 control points
# control_points = np.array([[-3,-2,-.5,1,0],[.5,6,5.5,3.7,2]]) # 5 control points
# control_points = np.array([[-3,-4,-2,-.5,1,0],[.5,3.5,6,5.5,3.7,2]]) # 6 control points
# control_points = np.array([[-3,-4,-2,-.5,1,0,2],[.5,3.5,6,5.5,3.7,2,-1]]) # 7 control points
control_points = np.array([[-3,-4,-2,-.5,1,0,2,3.5,3],[.5,3.5,6,5.5,3.7,2,-1,2,5]]) # figures
control_points = np.array([[-3,-4,-2,-.5,1,0,2,3.5,3],[.5,3.5,6,5.5,3.7,2,-1,2,5],[1,3.2,5,0,3.3,1.5,-1,2.5,4]]) # 3 D
# control_points = np.array([[0,0,0,0,0,2,3,5,6,7.4,8,9.5,10,10,10,10,10],[0,0,0,0,0,2,5,8,6,4.5,7,9.5,10,10,10,10,10]]) # Poster
# control_points = np.array([1,2,2.5,4,5.2,6,6.3,5]) # 1 dimensional
order = 3
start_time = 0
number_data_points = 1000
dimension = get_dimension(control_points)
scale_factor = 1
r = 3
clamped = True

bspline = BsplineEvaluation(control_points, order, start_time,scale_factor,clamped)

start = time.time()
spline_data, time_data = bspline.get_spline_data(number_data_points)
print("spline_data: " , np.shape(spline_data))
end = time.time()
print("Time function: " , end - start)
spline_derivative_data, time_data_2 = bspline.get_spline_derivative_data(number_data_points,r)
spline_curvature_data, time_data_3 = bspline.get_spline_curvature_data(number_data_points)
basis_function_data, time_data_4 = bspline.get_basis_function_data(number_data_points)


knot_points = bspline.get_knot_points()
defined_knot_points = bspline.get_defined_knot_points()
spline_at_knot_points = bspline.get_spline_at_knot_points()

print("control points: " , control_points)
print("knot_points: " , knot_points)
print("defined knot points: " , defined_knot_points)
print("spline at knots: " , spline_at_knot_points)


# bspline.plot_spline(number_data_points)
# bspline.plot_spline_vs_time(number_data_points)
# bspline.plot_basis_functions(number_data_points)
bspline.plot_derivative(number_data_points, r)

plt.figure("Curvature")
plt.plot(time_data_3, spline_curvature_data, color='red')
plt.xlabel('time')
plt.ylabel('curvature')
plt.title("Curvature")
# plt.legend()
plt.show()


