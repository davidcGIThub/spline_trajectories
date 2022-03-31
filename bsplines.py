"""
This module contains code to evaluate an open uniform b spline 
using the matrix method and the cox-de-boor table method for splines of order 
higher than the 5th degree. This also evaluates the derivatives of the B-spline
"""

import numpy as np 

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
        self._end_time = self._knot_points[self.__get_number_of_control_points()]

    def get_spline_data(self , number_of_data_points):
        '''
        Returns equally distributed data points for the spline, as well
        as time data for the parameterization
        '''
        time_data = np.linspace(self._start_time, self._end_time, number_of_data_points)
        dimension = self._control_points.ndim
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
        dimension = self._control_points.ndim
        if dimension == 1:
            spline_derivative_data = np.zeros(number_of_data_points)
        else:
            spline_derivative_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_derivative_data[i] = self.__evaluate_derivative_at_time_t(t,rth_derivative)
            else:
                spline_derivative_data[:,i][:,None] = self.__evaluate_derivative_at_time_t(t,rth_derivative)
        return spline_derivative_data, time_data

    def get_defined_knot_points(self):
        '''
        returns the knot points that are defined along the curve
        '''
        number_of_control_points = self.__get_number_of_control_points()
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
        dimension = self._control_points.ndim
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

    def get_spline_at_time_t(self, time):
        """
        This function evaluates the B spline at the given time
        """
        if self._order > 5 or (self._clamped and self._order > 3):
            # spline_at_time_t = self.__cox_de_boor_table_method(time)
            spline_at_time_t = self.__cox_de_boor_recursion_method(time)
        else:
            spline_at_time_t = self.__matrix_method(time)
        return spline_at_time_t


    def get_spline_derivative_at_time_t(self,time,rth_derivative):
        """
        This function evaluates the B spline derivative at the given time
        """
        if self._order > 5:
            derivative_at_time_t = self.__evaluate_derivative_finite_difference_method(time,rth_derivative)
        else:
            derivative_at_time_t = self.__evaluate_derivative_matrix_method(time, rth_derivative)
        return derivative_at_time_t

    def get_time_to_control_point_correlation(self):
        '''
        This is not a true correlation but distributes the control points
        evenly through the time interval and provides a time to each control point
        '''
        number_of_control_points = self.__get_number_of_control_points()
        time_array = np.linspace(self._start_time, self._end_time, number_of_control_points)
        return time_array

    def __create_knot_points(self):
        '''
        This function creates evenly distributed knot points
        '''
        number_of_control_points = self.__get_number_of_control_points()
        number_of_knot_points = number_of_control_points + self._order + 1
        knot_points = np.arange(number_of_knot_points)*self._scale_factor + self._start_time - self._order*self._scale_factor
        return knot_points

    def __create_clamped_knot_points(self):
        """ 
        Creates the list of knot points in the closed interval [t_0, t_{k+p}] 
        with the first k points equal to t_k and the last k points equal to t_{p}
        where k = order of the polynomial, and p = number of control points
        """
        number_of_control_points = self.__get_number_of_control_points()
        number_of_knot_points = number_of_control_points + self._order + 1
        number_of_unique_knot_points = number_of_knot_points - 2*self._order
        unique_knot_points = np.arange(0,number_of_unique_knot_points) * self._scale_factor + self._start_time
        knot_points = np.zeros(number_of_knot_points) + self._start_time
        knot_points[self._order : self._order + number_of_unique_knot_points] = unique_knot_points
        knot_points[self._order + number_of_unique_knot_points: 2*self._order + number_of_unique_knot_points] = unique_knot_points[-1]
        return knot_points

        
    def __find_preceding_knot_index(self,time):
        """ 
        This function finds the knot point preceding
        the current time
        """
        preceding_knot_index = -1
        number_of_control_points = self.__get_number_of_control_points()
        if time >= self._knot_points[number_of_control_points-1]:
            preceding_knot_index = number_of_control_points-1
        else:
            for knot_index in range(self._order,number_of_control_points+1):
                preceding_knot_index = number_of_control_points - 1
                knot_point = self._knot_points[knot_index]
                next_knot_point = self._knot_points[knot_index + 1]
                if time >= knot_point and time < next_knot_point:
                    preceding_knot_index = knot_index
                    break
        return preceding_knot_index

    def __get_number_of_control_points(self):
        if self._control_points.ndim == 1:
            number_of_control_points = len(self._control_points)
        else:
            number_of_control_points = len(self._control_points[0])
        return number_of_control_points

    def __cox_de_boor_recursion_method(self, time):
        """
        This function evaluates the B spline at the given time
        """
        preceding_knot_index = self.__find_preceding_knot_index(time)
        initial_control_point_index = preceding_knot_index - self._order
        dimension = self._control_points.ndim
        spline_at_time_t = np.zeros((dimension,1))
        for i in range(initial_control_point_index , initial_control_point_index+self._order + 1):
            if dimension == 1:
                control_point = self._control_points[i]
            else:
                control_point = self._control_points[:,i][:,None]
            basis_function = self.__cox_de_boor_recursion_basis_function(time, i, self._order)
            spline_at_time_t += basis_function*control_point
        return spline_at_time_t

    def __cox_de_boor_table_method(self, time):
        """
        This function evaluates the B spline at the given time
        """
        preceding_knot_index = self.__find_preceding_knot_index(time)
        initial_control_point_index = preceding_knot_index - self._order
        dimension = self._control_points.ndim
        spline_at_time_t = np.zeros((dimension,1))
        for i in range(initial_control_point_index , initial_control_point_index+self._order + 1):
            if dimension == 1:
                control_point = self._control_points[i]
            else:
                control_point = self._control_points[:,i][:,None]
            basis_function = self.__cox_de_boor_table_basis_function(time, i, self._order)
            spline_at_time_t += basis_function*control_point
        return spline_at_time_t

    def __cox_de_boor_table_basis_function(self, time, i, kappa):
        number_of_control_points = self.__get_number_of_control_points()
        table = np.zeros((self._order+1,self._order+1))
        #loop through rows to create the first column
        for y in range(self._order+1):
            t_i_y = self._knot_points[i+y]
            t_i_y_1 = self._knot_points[i+y+1]
            if time >= t_i_y and time < t_i_y_1:
                table[y,0] = 1
            elif time == t_i_y_1 and t_i_y_1 == self._knot_points[number_of_control_points] and self._clamped:
                table[y,0] = 1
        # loop through remaining columns
        for kappa in range(1,self._order+1): 
            # loop through rows
            number_of_rows = self._order+1 - kappa
            for y in range(number_of_rows):
                t_i_y = self._knot_points[i+y]
                t_i_y_1 = self._knot_points[i+y+1]
                t_i_y_k = self._knot_points[i+y+kappa]
                t_i_y_k_1 = self._knot_points[i+y+kappa+1]
                horizontal_term = 0
                diagonal_term = 0
                if t_i_y_k > t_i_y:
                    horizontal_term = table[y,kappa-1] * (time - t_i_y) / (t_i_y_k - t_i_y)
                if t_i_y_k_1 > t_i_y_1:
                    diagonal_term = table[y+1,kappa-1] * (t_i_y_k_1 - time)/(t_i_y_k_1 - t_i_y_1)
                table[y,kappa] = horizontal_term + diagonal_term
        return table[0,self._order]
    
    def __get_ith_control_point(self, i):
        dimension = self._control_points.ndim
        if dimension == 1:
            control_point = self._control_points[i]
        else:
            control_point = self._control_points[:,i][:,None]
        return control_point

    def __matrix_method(self, time):
        """
        This function evaluates the B spline at the given time
        """
        preceding_knot_index = self.__find_preceding_knot_index(time)
        preceding_knot_point = self._knot_points[preceding_knot_index]
        initial_control_point_index = preceding_knot_index - self._order
        dimension = self._control_points.ndim
        spline_at_time_t = np.zeros((dimension,1))
        M = self.__get_M_matrix(initial_control_point_index)
        i_p = initial_control_point_index
        tau = time - preceding_knot_point
        if dimension > 1:
            P = np.zeros((dimension,self._order+1))
        else:
            P = np.zeros(self._order+1)
        T = np.ones((self._order+1,1))
        for i in range(self._order+1):
            y = i
            kappa = i
            if dimension > 1:
                P[:,y] = self._control_points[:,i_p+y]
            else:
                P[y] = self._control_points[i_p+y]
            T[kappa,0] = (tau/self._scale_factor)**(self._order-kappa)
        spline_at_time_t = np.dot(P, np.dot(M,T))
        return spline_at_time_t

    def __get_M_matrix(self, initial_control_point_index):
        if self._order == 1:
            M = self.__get_1_order_matrix()
        if self._order == 2:
            M = self.__get_2_order_matrix(initial_control_point_index)
        elif self._order == 3:
            M = self.__get_3_order_matrix(initial_control_point_index)
        elif self._order == 4:
            M = self.__get_4_order_matrix(initial_control_point_index)
        elif self._order == 5:
            M = self.__get_5_order_matrix(initial_control_point_index)
        return M

    def __get_1_order_matrix(self):
        M = np.array([[-1,1],
                       [1,0]])
        return M

    def __get_2_order_matrix(self,initial_control_point_index):
        i_t = initial_control_point_index + self._order
        n = self.__get_number_of_control_points()
        M = .5*np.array([[1,-2,1],
                         [-2,2,1],
                         [1,0,0]])
        if self._clamped:
            if i_t == 2 and self._knot_points[2] < self._knot_points[n-1]:
                M = .5*np.array([[2,-4,2],
                                [-3,4,0],
                                [1,0,0]])
            elif i_t == n - 1 and self._knot_points[2] < self._knot_points[n-1]:
                M = .5*np.array([[1,-2,1],
                                [-3,2,1],
                                [2,0,0]])
            elif i_t == 2 and self._knot_points[2] == self._knot_points[n-1] :
                M = 0.5*np.array([[2,-4,2],
                                  [-4,4,0],
                                  [2,0,0]])
        return M

    def __get_3_order_matrix(self, initial_control_point_index):
        i_t = initial_control_point_index + self._order
        n = self.__get_number_of_control_points()
        M = np.array([[-2 ,  6 , -6 , 2],
                      [ 6 , -12 ,  0 , 8],
                      [-6 ,  6 ,  6 , 2],
                      [ 2 ,  0 ,  0 , 0]])/12
        if self._clamped:
            if i_t == 3 and self._knot_points[3] < self._knot_points[n-1]:
                M = np.array([[-12 ,  36 , -36 , 12],
                            [ 21 , -54 ,  36 , 0],
                            [-11 ,  18 ,  0  , 0],
                            [ 2  ,  0  ,  0  , 0]])/12.0
            elif i_t == 4 and self._knot_points[4] < self._knot_points[n-2]:
                M = np.array([[-3 ,  9 , -9 , 3],
                            [ 7 , -15 ,  3 , 7],
                            [-6 ,  6 , 6 , 2],
                            [ 2 ,  0 ,  0 , 0]])/12

            elif i_t == n - 2 and self._knot_points[4] < self._knot_points[n-2]:
                M = np.array([[-2  , 6   ,  -6 , 2],
                            [6 , -12  , 0 ,  8],
                            [ -7,    6 ,  6 , 2 ],
                            [ 3,  0   , 0  , 0]])/12
            elif i_t == n - 1 and self._knot_points[3] < self._knot_points[n-1]:
                M = np.array([[-2  , 6   ,  -6 , 2],
                            [11 , -15  , -3 ,  7],
                            [-21,  9   ,  9 ,  3],
                            [12 ,  0   ,  0 ,  0]])/12.0
            elif i_t == 4 and self._knot_points[4] == self._knot_points[n-2]:
                M = np.array([[-3 , 9 , -9 , 3],
                    [7, -15 , 3 , 7],
                    [-7 , 6 , 6 , 2],
                    [3 , 0 , 0 , 0]])/12
            elif i_t == 3 and self._knot_points[3] == self._knot_points[n-1]:
                 M = np.array([[-12 , 36 , -36 , 12],
                      [36 , -72 , 36 , 0],
                      [-36 , 36 , 0 , 0],
                      [12 , 0 , 0 , 0]])/12
        return M

    def __get_4_order_matrix(self, initial_control_point_index):
        M = np.array([[ 1 , -4  ,  6 , -4  , 1],
                      [-4 ,  12 , -6 , -12 , 11],
                      [ 6 , -12 , -6 ,  12 , 11],
                      [-4 ,  4  ,  6 ,  4  , 1],
                      [ 1 ,  0  ,  0 ,  0  , 0]])/24
        return M

    def __get_5_order_matrix(self, initial_control_point_index):
        M = np.array([[-1  ,  5  , -10 ,  10 , -5  , 1],
                      [ 5  , -20 ,  20 ,  20 , -50 , 26],
                      [-10 ,  30 ,  0  , -60 ,  0  , 66],
                      [ 10 , -20 , -20 ,  20 ,  50 , 26],
                      [-5  ,  5  ,  10 ,  10 ,  5  , 1 ],
                      [ 1  ,  0  ,  0  ,  0  ,  0  , 0]])/120
        return M

    def __evaluate_derivative_at_time_t(self, time, rth_derivative):
        '''
        This function evaluates the rth derivative of the spline at time t
        '''
        # if self._order > 5 or (self._order > 3 and self._clamped):
        #     derivative_at_time_t = self.__evaluate_rth_derivative_step_method(time,rth_derivative)
        # else:
        # derivative_at_time_t = self.__evaluate_rth_derivative_matrix_method(time, rth_derivative)
        derivative_at_time_t = self.__evaulate_derivative_at_time_t_recursive(time, rth_derivative)
        return derivative_at_time_t

    def __evaluate_rth_derivative_matrix_method(self, time, rth_derivative):
        preceding_knot_index = self.__find_preceding_knot_index(time)
        preceding_knot_point = self._knot_points[preceding_knot_index]
        initial_control_point_index = preceding_knot_index - self._order
        dimension = self._control_points.ndim
        M = self.__get_M_matrix(initial_control_point_index)
        i_p = initial_control_point_index
        tau = (time - preceding_knot_point)
        if dimension > 1:
            P = np.zeros((dimension,self._order+1))
        else:
            P = np.zeros(self._order+1)
        for y in range(self._order+1):
            if dimension > 1:
                P[:,y] = self._control_points[:,i_p+y]
            else:
                P[y] = self._control_points[i_p+y]
        T = np.zeros((self._order+1,1))
        for i in range(self._order-rth_derivative+1):
            T[i,0] = (tau**(self._order-rth_derivative-i)*np.math.factorial(self._order-i)) /  (self._scale_factor**(self._order-i)*np.math.factorial(self._order-i-rth_derivative))
        spline_derivative_at_time_t = np.dot(P, np.dot(M,T))
        return spline_derivative_at_time_t


    def __evaluate_rth_derivative_step_method(self, time, rth_derivative, delta = 1e-6):
        # min delta stuff
        #check if derivative is possible
        if rth_derivative == 1:
            return self.__evaluate_derivative_complex_step_method(time)
        else:
            dimension = self._control_points.ndim
            if time-delta*(rth_derivative-1) < self._start_time:
                # number_of_slots = rth_derivative
                derivative_time = self._start_time
                # derivative_index = int(np.floor(rth_derivative/2))
            elif time+delta*(rth_derivative-1) > self._end_time:
                # number_of_slots = rth_derivative
                derivative_time = self._end_time-delta*rth_derivative*2
                # derivative_index = int(np.ceil(rth_derivative/2))
            else:
                # number_of_slots = 2*rth_derivative-1
                derivative_time = time-delta*rth_derivative
            derivative_index = rth_derivative
            number_of_slots = 2*rth_derivative-1
            if dimension > 1:
                derivative_array = np.zeros((dimension, number_of_slots))
                axis = 1
            else:
                derivative_array = np.zeros(number_of_slots)
                axis = 0
            # fill derivative array
            for i in range(number_of_slots):
                if dimension == 1:
                    derivative_array[i] = self.__evaluate_derivative_complex_step_method(derivative_time)
                else:
                    derivative_array[:,i][:,None] = self.__evaluate_derivative_complex_step_method(derivative_time)
                derivative_time += delta
            # take the derivatives
            # print("derivative_array: " , derivative_array)
            for i in range(1,rth_derivative+1):
                derivative_array = np.gradient(derivative_array,delta,axis=axis)
                # derivative_array = np.diff(derivative_array,axis)/delta
                if i > self._order:
                    derivative_array = derivative_array*0
                    break
            if dimension == 1:
                derivative = derivative_array[derivative_index]
            else:
                derivative = derivative_array[:,derivative_index][:,None]
            return derivative

    def __evaluate_derivative_complex_step_method(self, time):
        delta = 1e-30
        complex_variable = time + delta*1j
        complex_function_output = self.get_spline_at_time_t(complex_variable)
        derivative_at_time_t = complex_function_output.imag / delta
        return derivative_at_time_t

    def __evaulate_derivative_at_time_t_recursive(self, time, r):
        """
        This function evaluates the B spline at the given time
        """
        preceding_knot_index = self.__find_preceding_knot_index(time)
        initial_control_point_index = preceding_knot_index - self._order
        dimension = self._control_points.ndim
        derivative_at_time_t = np.zeros((dimension,1))
        for i in range(initial_control_point_index , initial_control_point_index+self._order + 1):
            if dimension == 1:
                control_point = self._control_points[i]
            else:
                control_point = self._control_points[:,i][:,None]
            iteration_value = self.__rth_derivative_cox_de_boor_recursion_basis_function(time, i,self._order, r) * control_point
            derivative_at_time_t += iteration_value
        return derivative_at_time_t
    
    def __cox_de_boor_recursion_basis_function(self, time, i, kappa):
        t = time
        t_i = self._knot_points[i]
        t_i_1 = self._knot_points[i+1]
        t_i_k = self._knot_points[i+kappa]
        t_i_k_1 = self._knot_points[i+kappa+1]
        if kappa == 0:
            if (t >= t_i and t < t_i_1) or (t == self._end_time and t >= t_i and t == t_i_1):
                basis_function = 1
            else:
                basis_function = 0
        else:
            if t_i < t_i_k:
                term1 = (t - t_i) / (t_i_k - t_i) * self.__cox_de_boor_recursion_basis_function(t,i,kappa-1)
            else:
                term1 = 0
            if t_i_1 < t_i_k_1:
                term2 = (t_i_k_1 - t)/(t_i_k_1 - t_i_1) * self.__cox_de_boor_recursion_basis_function(t,i+1,kappa-1)
            else:
                term2 = 0
            basis_function = term1 + term2
        return basis_function

    def __derivative_cox_de_boor_recursion_basis_function(self, time, i, kappa):
        t = time
        t_i = self._knot_points[i]
        t_i_1 = self._knot_points[i+1]
        t_i_k = self._knot_points[i+kappa]
        t_i_k_1 = self._knot_points[i+kappa+1]
        if kappa == 0:
            if t >= t_i and t < t_i_1:
                basis_function = 1
            else:
                basis_function = 0
        else:
            if t_i < t_i_k:
                term1 = (1) / (t_i_k - t_i) * self.__cox_de_boor_recursion_basis_function(t,i,kappa-1)
                if kappa-1 > 0:
                    term1 += (t - t_i) / (t_i_k - t_i) * self.__derivative_cox_de_boor_recursion_basis_function(t,i,kappa-1)
            else:
                term1 = 0
            if t_i_1 < t_i_k_1:
                term2 = (-1)/(t_i_k_1 - t_i_1) * self.__cox_de_boor_recursion_basis_function(t,i+1,kappa-1)
                if kappa-1 > 0:
                    term2 += (t_i_k_1 - t)/(t_i_k_1 - t_i_1) * self.__derivative_cox_de_boor_recursion_basis_function(t,i+1,kappa-1)
            else:
                term2 = 0
            basis_function = term1 + term2
        return basis_function


    def __rth_derivative_cox_de_boor_recursion_basis_function(self, time, i, kappa, r):
        if r == 0:
            return self.__cox_de_boor_recursion_basis_function(time, i, kappa)
        t = time
        t_i = self._knot_points[i]
        t_i_1 = self._knot_points[i+1]
        t_i_k = self._knot_points[i+kappa]
        t_i_k_1 = self._knot_points[i+kappa+1]
        if kappa == 0:
            # if t >= t_i and t < t_i_1:
            #     basis_function = 1
            # else:
            basis_function = 0
        else:
            if t_i < t_i_k:
                if kappa-1 > 0:
                    term1 = (t - t_i) / (t_i_k - t_i) * self.__rth_derivative_cox_de_boor_recursion_basis_function(t,i,kappa-1,r) + \
                        (r) / (t_i_k - t_i) * self.__rth_derivative_cox_de_boor_recursion_basis_function(t,i,kappa-1, r-1)
                else:
                    term1 = (1) / (t_i_k - t_i) * self.__rth_derivative_cox_de_boor_recursion_basis_function(t,i,kappa-1, r-1)
            else:
                term1 = 0
            if t_i_1 < t_i_k_1:
                if kappa-1 > 0:
                    term2 = (t_i_k_1 - t)/(t_i_k_1 - t_i_1) * self.__rth_derivative_cox_de_boor_recursion_basis_function(t,i+1,kappa-1,r) + \
                        (-r)/(t_i_k_1 - t_i_1) * self.__rth_derivative_cox_de_boor_recursion_basis_function(t,i+1,kappa-1,r-1)
                else:
                    term2 = (-1)/(t_i_k_1 - t_i_1) * self.__rth_derivative_cox_de_boor_recursion_basis_function(t,i+1,kappa-1,r-1)
            else:
                term2 = 0
            basis_function = term1 + term2
        return basis_function
