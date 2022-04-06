"""
This module contains code to evaluate an open uniform b spline 
using the matrix method and the cox-de-boor table method for splines of order 
higher than the 5th degree. This also evaluates the derivatives of the B-spline
"""

import numpy as np 
from matrix_evaluation import matrix_bspline_evaluation, derivative_matrix_bspline_evaluation
from table_evaluation import table_bspline_evaluation, derivative_table_bspline_evaluation
from helper_functions import count_number_of_control_points, get_dimension
class BsplineEvaluation:
    """
    This class contains contains code to evaluate an open uniform b spline 
    using the matrix method and the cox-de-boor table method for splines of order
    higher than the 5th degree. This also uses the table method for clamped B-splines
    of order higher than 3. This also evaluates the derivatives of the B-spline.
    """

    def __init__(self, control_points, order, start_time, scale_factor=1, clamped=False):
        '''
        Constructor for the BsplinEvaluation class, each column of
        control_points is a control point. Start time should be an integer.
        '''
        self._control_points = control_points
        self._order = order
        self._scale_factor = scale_factor
        self._start_time = start_time
        self._clamped = clamped
        if clamped:
            self._knot_points = self.__create_clamped_knot_points()
        else:
            self._knot_points = self.__create_knot_points()
        self._end_time = self._knot_points[count_number_of_control_points(self._control_points)]

    def get_spline_data(self , number_of_data_points):
        '''
        Returns equally distributed data points for the spline, as well
        as time data for the parameterization
        '''
        time_data = np.linspace(self._start_time, self._end_time, number_of_data_points)
        dimension = get_dimension(self._control_points)
        if dimension == 1:
            spline_data = np.zeros(number_of_data_points)
        else:
            spline_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_data[i] = self.get_spline_at_time_t(t)
            else:
                spline_data[:,i][:,None] = self.get_spline_at_time_t(t)
        return spline_data, time_data

    def get_spline_derivative_data(self,number_of_data_points, rth_derivative):
        '''
        Returns equally distributed data points for the derivative of the spline, 
        as well as time data for the parameterization
        '''
        time_data = np.linspace(self._start_time, self._end_time, number_of_data_points)
        dimension = get_dimension(self._control_points)
        if dimension == 1:
            spline_derivative_data = np.zeros(number_of_data_points)
        else:
            spline_derivative_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_derivative_data[i] = self.get_derivative_at_time_t(t, rth_derivative)
            else:
                spline_derivative_data[:,i][:,None] = self.get_derivative_at_time_t(t, rth_derivative)
        return spline_derivative_data, time_data

    def get_spline_curvature_data(self,number_of_data_points):
        '''
        Returns equally distributed data points for the curvature of the spline, 
        as well as time data for the parameterization
        '''
        time_data = np.linspace(self._start_time, self._end_time, number_of_data_points)
        spline_curvature_data = np.zeros(number_of_data_points)
        for i in range(number_of_data_points):
            t = time_data[i]
            spline_curvature_data[i] = self.get_curvature_at_time_t(t)
        return spline_curvature_data, time_data


    def get_spline_at_time_t(self, time):
        """
        This function evaluates the B spline at the given time
        """
        if self._order > 5 or (self._clamped and self._order > 3):
            spline_at_time_t = table_bspline_evaluation(time, self._control_points, self._knot_points, self._clamped)
        else:
            spline_at_time_t = matrix_bspline_evaluation(time, self._scale_factor, self._control_points, self._knot_points, self._clamped)
        return spline_at_time_t

    def get_derivative_at_time_t(self, time, derivative_order):
        '''
        This function evaluates the rth derivative of the spline at time t
        '''
        if self._order > 5 or (self._order > 3 and self._clamped):
            derivative_at_time_t = derivative_table_bspline_evaluation(time, derivative_order, self._control_points, self._knot_points, self._clamped)       
        else:
            derivative_at_time_t = derivative_matrix_bspline_evaluation(time, derivative_order, self._scale_factor, self._control_points, self._knot_points, self._clamped)
        return derivative_at_time_t

    def get_curvature_at_time_t(self, time):
        '''
        This function evaluates the curvature at time t
        '''
        dimension = get_dimension(self._control_points)
        if dimension == 1:
            derivative_vector = np.array([1 , self.get_derivative_at_time_t(time,1)[0]])
            derivative_2nd_vector = np.array([0 , self.get_derivative_at_time_t(time,2)[0]])
        else:
            derivative_vector = self.get_derivative_at_time_t(time,1)
            derivative_2nd_vector = self.get_derivative_at_time_t(time,2)
        curvature = np.linalg.norm(np.cross(derivative_vector.flatten(), derivative_2nd_vector.flatten())) / np.linalg.norm(derivative_vector)**3
        return curvature

    def get_defined_knot_points(self):
        '''
        returns the knot points that are defined along the curve
        '''
        number_of_control_points = count_number_of_control_points(self._control_points)
        defined_knot_points = self._knot_points[self._order:number_of_control_points]
        return defined_knot_points

    def get_knot_points(self):
        '''
        returns all the knot points
        '''
        return self._knot_points

    def get_spline_at_knot_points(self):
        '''
        Returns spline data evaluated at the knot points for
        which the spline is defined.
        '''
        time_data = self.get_defined_knot_points()
        number_of_data_points = len(time_data)
        dimension = get_dimension(self._control_points)
        if dimension == 1:
            spline_data = np.zeros(number_of_data_points)
        else:
            spline_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_data[i] = self.get_spline_at_time_t(t)
            else:
                spline_data[:,i][:,None] = self.get_spline_at_time_t(t)
        return spline_data

    def get_time_to_control_point_correlation(self):
        '''
        This is not a true correlation but distributes the control points
        evenly through the time interval and provides a time to each control point
        '''
        number_of_control_points = count_number_of_control_points(self._control_points)
        time_array = np.linspace(self._start_time, self._end_time, number_of_control_points)
        return time_array

    def __create_knot_points(self):
        '''
        This function creates evenly distributed knot points
        '''
        number_of_control_points = count_number_of_control_points(self._control_points)
        number_of_knot_points = number_of_control_points + self._order + 1
        knot_points = np.arange(number_of_knot_points)*self._scale_factor + self._start_time - self._order*self._scale_factor
        return knot_points

    def __create_clamped_knot_points(self):
        """ 
        Creates the list of knot points in the closed interval [t_0, t_{k+p}] 
        with the first k points equal to t_k and the last k points equal to t_{p}
        where k = order of the polynomial, and p = number of control points
        """
        number_of_control_points = count_number_of_control_points(self._control_points)
        number_of_knot_points = number_of_control_points + self._order + 1
        number_of_unique_knot_points = number_of_knot_points - 2*self._order
        unique_knot_points = np.arange(0,number_of_unique_knot_points) * self._scale_factor + self._start_time
        knot_points = np.zeros(number_of_knot_points) + self._start_time
        knot_points[self._order : self._order + number_of_unique_knot_points] = unique_knot_points
        knot_points[self._order + number_of_unique_knot_points: 2*self._order + number_of_unique_knot_points] = unique_knot_points[-1]
        return knot_points
