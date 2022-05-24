"""
This module contains code to evaluate an piecewise uniform b splines 
This also evaluates the derivatives of the B-spline
"""

import numpy as np 
import matplotlib.pyplot as plt
from bsplinegenerator.bsplines import BsplineEvaluation

class PiecewiseBsplineEvaluation:
    """
    This class contains code to evaluate piecewise uniform b splines 
    This also evaluates the derivatives of the B-spline
    """
    def __init__(self, order, control_points_list, scale_factor_list, start_time = 0, clamped = False):
        '''
        Constructor for the BsplinEvaluation class, each column of
        control_points is a control point. Start time should be an integer.
        '''
        self._order = order
        self._control_point_list = control_points_list
        self._scale_factor_list = scale_factor_list
        self._start_time = start_time
        self._clamped = clamped
        self._bspline_list = self.__create_bspline_list(self._order, self._control_point_list, 
            self._scale_factor_list, self._start_time, self._clamped)

    def get_spline_data(self, number_of_data_points):
        '''
        Returns equally distributed data points for the piecewise spline, as well
        as time data for the parameterization
        '''
        end_time = self.get_end_time()
        time_data = np.linspace(self._start_time, end_time , number_of_data_points)
        dimension = self.__get_dimension()
        if dimension == 1:
            spline_data = np.zeros(number_of_data_points)
        else:
            spline_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_data[i] = self.get_spline_at_time(t)
            else:
                spline_data[:,i][:,None] = self.get_spline_at_time(t)
        return spline_data, time_data

    def get_spline_derivative_data(self, number_of_data_points, derivative_order=1):
        '''
        Returns equally distributed data points for the derivative of
        the piecewise spline, as well as time data for the parameterization
        '''
        end_time = self.get_end_time()
        time_data = np.linspace(self._start_time, end_time , number_of_data_points)
        dimension = self.__get_dimension()
        if dimension == 1:
            spline_derivative_data = np.zeros(number_of_data_points)
        else:
            spline_derivative_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_derivative_data[i] = self.get_spline_derivative_at_time(t)
            else:
                spline_derivative_data[:,i][:,None] = self.get_spline_derivative_at_time(t)
        return spline_derivative_data, time_data

    def get_spline_at_time(self, time):
        spline_number = self.__get_spline_number_at_time(time)
        spline_at_time = self._bspline_list[spline_number].get_spline_at_time_t(time)
        return spline_at_time

    def get_spline_derivative_at_time(self, time, derivative_order):
        spline_number = self.__get_spline_number_at_time(time)
        spline_derivative_at_time = self._bspline_list[spline_number].get_derivative_at_time_t(time, derivative_order)
        return spline_derivative_at_time

    def __get_terminal_knot_points(self):
        num_terminal_knot_points =  self.__get_number_of_terminal_knot_points()
        terminal_knot_points = np.zeros(num_terminal_knot_points)
        time_count = self._start_time
        for i in range(num_terminal_knot_points):
            terminal_knot_points[i] = time_count
            if i < self.__get_number_of_splines():
                num_control_points = self.__get_num_control_points_for_spline(i)
                num_intervals = num_control_points - self._order
                scale_factor = self._scale_factor_list[i]
                time_count += scale_factor*num_intervals
        return terminal_knot_points

    # def get_curvature_at_time(self, time):
    #     spline_number = self.__get_spline_number_at_time(time)
    #     spline_curvature_at_time = self._bspline_list[spline_number].get_curvature_at_time_t(time)
    #     return spline_curvature_at_time

    def get_start_time(self):
        return self._start_time

    def get_end_time(self):
        num_splines = self.__get_number_of_splines()
        return self._bspline_list[num_splines-1].get_end_time()

    def __get_spline_number_at_time(self, time):
        time_count = self._start_time
        spline_number = 0
        num_splines = self.__get_number_of_splines()
        for i in range(num_splines):
            scale_factor = self._scale_factor_list[i]
            num_control_points = self.__get_num_control_points_for_spline(i)
            num_intervals = num_control_points - self._order
            if time < time_count+scale_factor*num_intervals and time >= time_count:
                spline_number = i
                break
            elif time == self.get_end_time():
                spline_number = num_splines - 1
            time_count += scale_factor*num_intervals
        return spline_number

    def __get_number_of_splines(self):
        return len(self._scale_factor_list)

    def __get_number_of_terminal_knot_points(self):
        return self.__get_number_of_splines() + 1
    
    def __get_num_control_points_for_spline(self, spline_number):
        dimension = self.__get_dimension()
        control_points = self._control_point_list[spline_number]
        if dimension == 1:
            num_control_points = len(control_points)
        else:
            num_control_points = np.shape(control_points)[1]
        return num_control_points

    def __create_bspline_list(self, order, control_points_list, scale_factor_list, start_time, clamped):
        number_of_bsplines = len(scale_factor_list)
        bspline_list = []
        spline_start_time = start_time
        for i in range(number_of_bsplines):
            control_points = control_points_list[i]
            num_control_points = self.__get_num_control_points_for_spline(i)
            num_intervals = num_control_points - order
            scale_factor = scale_factor_list[i]
            bspline = BsplineEvaluation(control_points, order, spline_start_time, scale_factor, clamped)
            bspline_list.append(bspline)
            spline_start_time += scale_factor*num_intervals
        return bspline_list

    def __get_dimension(self):
        control_points = self._control_point_list[0]
        if control_points.ndim == 1:
            dimension = 1
        else:
            dimension = len(control_points)
        return dimension

    def plot_splines(self, number_of_data_points, show_control_points = True, show_terminal_knot_points = True):
        figure_title = str(self._order) + " Order Piecewise B-Spline"
        dimension = self.__get_dimension()
        plt.figure(figure_title)
        if dimension == 3:
            self.__plot_3d_splines(number_of_data_points, show_control_points, show_terminal_knot_points)
        elif dimension == 2:
            self.__plot_2d_splines(number_of_data_points, show_control_points, show_terminal_knot_points)
        elif dimension == 1:
            self.__plot_1d_splines(number_of_data_points, show_control_points, show_terminal_knot_points)
        else:
            print("Spline dimensions to high to show representation")
        plt.title(figure_title)
        plt.legend()
        plt.show()
    
    def __plot_3d_splines(self, number_of_data_points, show_control_points, show_terminal_knot_points):
        spline_data, time_data = self.get_spline_data(number_of_data_points)
        terminal_knot_points = self.__get_terminal_knot_points()
        num_splines = self.__get_number_of_splines()
        ax = plt.axes(projection='3d')
        ax.set_box_aspect(aspect =(1,1,1))
        for i in range(num_splines):
            start_time = terminal_knot_points[i]
            end_time = terminal_knot_points[i+1]
            start_index = np.argmin(np.abs(time_data-start_time))+1
            end_index = np.argmin(np.abs(time_data-end_time))
            # plot control points
            if (show_control_points):
                control_points = self._control_point_list[i]
                ax.plot(control_points[0,:], control_points[1,:],control_points[2,:],color="orange")
                if i == 0:
                    ax.scatter(control_points[0,:], control_points[1,:],control_points[2,:],color="orange",label="Control Points")
                else:
                    ax.scatter(control_points[0,:], control_points[1,:],control_points[2,:],color="orange")
            # plot terminal knot points
            if (show_terminal_knot_points):
                if i == 0:
                    ax.scatter(spline_data[0,start_index], spline_data[1,start_index],spline_data[2,start_index],color="blue",label="Spline at Terminal Knot Points")
                else:
                    ax.scatter(spline_data[0,start_index], spline_data[1,start_index],spline_data[2,start_index],color="blue")
                ax.scatter(spline_data[0,end_index], spline_data[1,end_index],spline_data[2,end_index],color="blue")
            # plot splines
            if i == 0:
                ax.plot(spline_data[0,start_index:end_index], spline_data[1,start_index:end_index],spline_data[2,start_index:end_index],color="blue",label="B-Spline")
            else:
                ax.plot(spline_data[0,start_index:end_index], spline_data[1,start_index:end_index],spline_data[2,start_index:end_index],color="blue")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def __plot_2d_splines(self, number_of_data_points, show_control_points, show_terminal_knot_points):
        spline_data, time_data = self.get_spline_data(number_of_data_points)
        terminal_knot_points = self.__get_terminal_knot_points()
        num_splines = self.__get_number_of_splines()
        for i in range(num_splines):
            start_time = terminal_knot_points[i]
            end_time = terminal_knot_points[i+1]
            start_index = np.argmin(np.abs(time_data-start_time))+1
            end_index = np.argmin(np.abs(time_data-end_time))
            #plot control points
            if (show_control_points):
                control_points = self._control_point_list[i]
                plt.plot(control_points[0,:], control_points[1,:],color="orange")
                if i == 0:
                    plt.scatter(control_points[0,:], control_points[1,:],linewidths=2,color="orange",label="Control Points")
                else:
                    plt.scatter(control_points[0,:], control_points[1,:],linewidths=2,color="orange")
            # plot terminal knot points
            if (show_terminal_knot_points):
                if i == 0:
                    plt.scatter(spline_data[0,start_index],spline_data[1,start_index],color='blue',label="Spline at Terminal Knot Points")
                else:
                    plt.scatter(spline_data[0,start_index],spline_data[1,start_index],color='blue')
                plt.scatter(spline_data[0,end_index],spline_data[1,end_index],color='blue')
            # plot splines
            if i == 0:
                plt.plot(spline_data[0,start_index:end_index], spline_data[1,start_index:end_index],color='blue',label="B-Spline")
            else:
                plt.plot(spline_data[0,start_index:end_index], spline_data[1,start_index:end_index],color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        ax = plt.gca()

    def __plot_1d_splines(self, number_of_data_points, show_control_points, show_terminal_knot_points):
        spline_data, time_data = self.get_spline_data(number_of_data_points)
        terminal_knot_points = self.__get_terminal_knot_points()
        num_splines = self.__get_number_of_splines()
        for i in range(num_splines):
            start_time = terminal_knot_points[i]
            end_time = terminal_knot_points[i+1]
            start_index = np.argmin(np.abs(time_data-start_time))+1
            end_index = np.argmin(np.abs(time_data-end_time))
            #plot control points
            if (show_control_points):
                control_points = self._control_point_list[i]
                control_point_times = np.linspace(start_time,end_time,len(control_points))
                plt.plot(control_point_times, control_points,color="orange")
                if i == 0:
                    plt.scatter(control_point_times, control_points,linewidths=2,color="orange",label="Control Points")
                else:
                    plt.scatter(control_point_times, control_points,linewidths=2,color="orange")
            # plot terminal knot points
            if (show_terminal_knot_points):
                if i == 0:
                    plt.scatter(time_data[start_index],spline_data[start_index],color='blue',label="Spline at Terminal Knot Points")
                else:
                    plt.scatter(time_data[start_index],spline_data[start_index],color='blue')
                plt.scatter(time_data[end_index],spline_data[end_index],color='blue')
            # plot splines
            if i == 0:
                plt.plot(time_data[start_index:end_index], spline_data[start_index:end_index],color='blue',label="B-Spline")
            else:
                plt.plot(time_data[start_index:end_index], spline_data[start_index:end_index],color='blue')
        plt.xlabel('time')
        plt.ylabel('b(t)')
        ax = plt.gca()
            
            # if (show_control_points):
            #     control_point_times = self.get_time_to_control_point_correlation()
            #     plt.scatter(control_point_times,control_points)
            #     plt.plot(control_point_times,control_points,label="Control Points")


#     def plot_spline_vs_time(self, number_of_data_points,show_knot_points = True):
#             figure_title = str(self._order) + " Order B-Spline vs Time"
#             dimension = get_dimension(self._control_points)
#             spline_data, time_data = self.get_spline_data(number_of_data_points)
#             spline_at_knot_points, defined_knot_points = self.get_spline_at_knot_points()
#             plt.figure(figure_title)
#             if(dimension > 1):
#                 for i in range(dimension):
#                     spline_label = "Dimension " + str(i)
#                     plt.plot(time_data, spline_data[i,:],label=spline_label)
#                     if (show_knot_points):
#                         plt.scatter(defined_knot_points, spline_at_knot_points[i,:])
#             else:
#                 plt.plot(time_data, spline_data,label="Spline")
#                 if (show_knot_points):
#                     plt.scatter(defined_knot_points, spline_at_knot_points)
#             plt.xlabel('time')
#             plt.ylabel('b(t)')
#             ax = plt.gca()
#             plt.title(figure_title)
#             plt.legend()
#             plt.show()


#     def plot_derivative(self, number_of_data_points, derivative_order):
#         figure_title = str(derivative_order) + " Order Derivative"
#         dimension = get_dimension(self._control_points)
#         spline_derivative_data, time_data = self.get_spline_derivative_data(number_of_data_points,derivative_order)
#         plt.figure(figure_title)
#         if dimension > 1:
#             for i in range(dimension):
#                 spline_label = "Dimension " + str(i)
#                 plt.plot(time_data, spline_derivative_data[i,:],label=spline_label)
#         else:
#             plt.plot(time_data, spline_derivative_data, label="Spline Derivative")
#         plt.xlabel('time')
#         plt.ylabel(str(derivative_order) + ' derivative')
#         plt.title(figure_title)
#         plt.legend()
#         plt.show()

#     def plot_curvature(self, number_of_data_points):
#         spline_curvature_data, time_data = self.get_spline_curvature_data(number_of_data_points)
#         plt.figure("Curvature")
#         plt.plot(time_data, spline_curvature_data)
#         plt.xlabel('time')
#         plt.ylabel('curvature')
#         plt.title("Curvature")
#         plt.show()
