import numpy as np
import time
import matplotlib.pyplot as plt
from helper_functions import get_dimension
from bsplines import BsplineEvaluation

# control_points = np.array([[0,3,-2,-.5,1,0,2,3,3,3],[0,4,6,5.5,3.7,2,-1,5,5,5]]) # initial velocity of 25
# control_points = np.random.randint(10, size=(2,13)) # random
# control_points = np.concatenate((np.array([[0,0,0,0],[0,0,0,0]]) , control_points , np.array([[10,10,10,10],[10,10,10,10]])),axis=1)
# control_points = np.array([[-3,-3,-3,-2,-.5,1,0,2,3,3,3],[.5,.5,.5,6,5.5,3.7,2,-1,5,5,5]]) # zero initial velocity
# control_points = np.array([[-.5,1,0],[5.5,3.7,2]]) # 3 control points
# control_points = np.array([[-2,-.5,1,0],[6,5.5,3.7,2]]) # 4 control points
# control_points = np.array([[-3,-2,-.5,1,0],[.5,6,5.5,3.7,2]]) # 5 control points
# control_points = np.array([[-3,-4,-2,-.5,1,0],[.5,3.5,6,5.5,3.7,2]]) # 6 control points
# control_points = np.array([[-3,-4,-2,-.5,1,0,2],[.5,3.5,6,5.5,3.7,2,-1]]) # 7 control points
control_points = np.array([[-3,-4,-2,-.5,1,0,2,3.5,3],[.5,3.5,6,5.5,3.7,2,-1,2,5]]) # figures
# control_points = np.array([[-3,-4,-2,-.5,1,0,2,3.5,3],[.5,3.5,6,5.5,3.7,2,-1,2,5],[1,3.2,5,0,3.3,1.5,-1,2.5,4]]) # 3 D
# control_points = np.array([[0,0,0,0,0,2,3,5,6,7.4,8,9.5,10,10,10,10,10],[0,0,0,0,0,2,5,8,6,4.5,7,9.5,10,10,10,10,10]]) # Poster
# control_points = np.array([1,2,2.5,4,5.2,6,6.3,5]) # 1 dimensional
order = 3
start_time = 0
number_data_points = 1000
dimension = get_dimension(control_points)
scale_factor = 1
r = 1
clamped = False

bspline = BsplineEvaluation(control_points, order, start_time,scale_factor,clamped)

start = time.time()
spline_data, time_data = bspline.get_spline_data(number_data_points)
print("spline_data: " , np.shape(spline_data))
end = time.time()
print("Time function: " , end - start)
spline_derivative_data, time_data_2 = bspline.get_spline_derivative_data(number_data_points,r)
spline_curvature_data, time_data_3 = bspline.get_spline_curvature_data(number_data_points)

knot_points = bspline.get_knot_points()
defined_knot_points = bspline.get_defined_knot_points()
spline_at_knot_points = bspline.get_spline_at_knot_points()

print("control points: " , control_points)
print("knot_points: " , knot_points)
print("defined knot points: " , defined_knot_points)
print("spline at knots: " , spline_at_knot_points)

plt.figure("B-Spline")
if dimension == 3:
    ax = plt.axes(projection='3d')
    # ax.set_aspect('auto')
    ax.set_box_aspect(aspect =(1,1,1))
    ax.plot(control_points[0,:], control_points[1,:],control_points[2,:],color='tab:orange')
    ax.scatter(control_points[0,:], control_points[1,:],control_points[2,:],color='tab:orange',label="control points")
    ax.plot(spline_data[0,:], spline_data[1,:],spline_data[2,:],label="B-spline")
    ax.scatter(spline_at_knot_points[0,:], spline_at_knot_points[1,:],spline_at_knot_points[2,:],label="B-spline at Knot Points")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
elif dimension == 2:
    plt.plot(control_points[0,:], control_points[1,:],color='tab:orange')
    plt.scatter(control_points[0,:], control_points[1,:],facecolors='none',edgecolors='tab:orange',linewidths=2,label="control points")
    plt.plot(spline_data[0,:], spline_data[1,:],label="B-spline")
    plt.scatter(spline_at_knot_points[0,:], spline_at_knot_points[1,:],label="B-spline at Knot Points")
    plt.xlabel('x')
    plt.ylabel('y')
    ax = plt.gca()
else:
    plt.scatter(bspline.get_time_to_control_point_correlation(),control_points,facecolors='none',edgecolors='tab:orange',linewidths=2,label="control points")
    plt.plot(time_data, spline_data,label="B-spline")
    plt.scatter(defined_knot_points, spline_at_knot_points,label="B-spline at Knot Points")
    plt.xlabel('time')
    plt.ylabel('b(t)')
    ax = plt.gca()
plt.title("B-Spline")
plt.legend()
plt.show()

if dimension == 3:
    x_derivative  = spline_derivative_data[0,:]
    y_derivative = spline_derivative_data[1,:]
    z_derivative = spline_derivative_data[2,:]
    plt.figure("Derivative")
    plt.plot(time_data, x_derivative, color='red')
    plt.plot(time_data, y_derivative, color='blue')
    plt.plot(time_data, z_derivative, color='green')
    plt.xlabel('time')
    plt.ylabel('derivative')
    plt.title("derivative")
    plt.legend()
    plt.show()
elif dimension == 2:
    x_derivative  = spline_derivative_data[0,:]
    y_derivative = spline_derivative_data[1,:]
    plt.figure("Derivative")
    plt.plot(time_data, x_derivative, color='red')
    plt.plot(time_data, y_derivative, color='blue')
    plt.xlabel('time')
    plt.ylabel('derivative')
    plt.title("derivative")
    plt.legend()
    plt.show()
else:
    plt.figure("Derivative")
    plt.plot(time_data, spline_derivative_data, color='red')
    plt.xlabel('time')
    plt.ylabel('derivative')
    plt.title("derivative")
    plt.legend()
    plt.show()


plt.figure("Curvature")
plt.plot(time_data_3, spline_curvature_data, color='red')
plt.xlabel('time')
plt.ylabel('curvature')
plt.title("Curvature")
# plt.legend()
plt.show()


