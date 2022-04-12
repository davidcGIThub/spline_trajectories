import numpy as np
import matplotlib.pyplot as plt



def plot_spline(spline_data, control_points, knot_points):
    figure_title = 
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
    elif dimension == 1:
        plt.scatter(bspline.get_time_to_control_point_correlation(),control_points,facecolors='none',edgecolors='tab:orange',linewidths=2,label="control points")
        plt.plot(time_data, spline_data,label="B-spline")
        plt.scatter(defined_knot_points, spline_at_knot_points,label="B-spline at Knot Points")
        plt.xlabel('time')
        plt.ylabel('b(t)')
        ax = plt.gca()
    else:
        #TODO - plot on the same axes
    plt.title("B-Spline")
    plt.legend()
    plt.show()


plt.figure("Basis Functions")
for b in range(count_number_of_control_points(control_points)):
    basis_function  = basis_function_data[b,:]
    plt.plot(time_data, basis_function)
plt.xlabel('time')
plt.ylabel('N(t)')
plt.title("Basis Functions")
plt.legend()
plt.show()

plt.figure("Derivative")
if dimension == 3:
    x_derivative  = spline_derivative_data[0,:]
    y_derivative = spline_derivative_data[1,:]
    z_derivative = spline_derivative_data[2,:]
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
    plt.plot(time_data, x_derivative, color='red')
    plt.plot(time_data, y_derivative, color='blue')
    plt.xlabel('time')
    plt.ylabel('derivative')
    plt.title("derivative")
    plt.legend()
    plt.show()
else:
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