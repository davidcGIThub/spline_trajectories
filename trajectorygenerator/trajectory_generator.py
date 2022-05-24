"""
This module contains code to generate B-spline trajectories with constraints
over the position of the trajectory and it's derivatives. It may also constrain
the curvature as well as region avoidance. ** currently works for dimension 2-3, and order***
"""

from tkinter import W
import numpy as np
from pyrsistent import v 
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from bsplinegenerator.matrix_evaluation import get_M_matrix, get_T_vector
from bsplinegenerator.bsplines

class TrajectoryGenerator:
    """
    This module contains code to generate B-spline trajectories through some waypoints
    with constraints over the derivatives of the trajectory, and combined derivative magnitudes.
    It may also constrain the curvature as well as region avoidance.
    """

    def __init__(self, order, interval_spacing_length = 1):
        # private variables
        self._interval_spacing_length = interval_spacing_length
        self._order = order
        self._control_point_list = []
        self._scale_factor_list = []
        
        # global variables used in obective and constraint functions
        self._dimension = 0
        self._number_of_splines = 0
        self._number_of_intervals_per_spline = np.array([])
        self._spline_first_control_point_indices = np.array([])
        

    def generate_trajectory(self, waypoints):
        # create initial conditions
        self._dimension = np.shape(waypoints)[0]
        self._number_of_splines = self.__get_number_of_splines(waypoints)
        self._number_of_intervals_per_spline = self.__get_number_of_intervals_per_spline_array(waypoints)
        self._spline_first_control_point_indices = self.__get_spline_first_control_point_indices_array(waypoints)
        num_splines = self.__get_number_of_splines(waypoints)
        initial_control_points = self.__create_initial_set_of_control_points(waypoints)
        initial_scale_factors = self.__create_initial_scale_factors(num_splines)
        optimization_variables = np.concatenate((initial_control_points.flatten(),initial_scale_factors))
        # define constraints and objective function
        waypoint_constraint = self.__create_waypoint_constraint(waypoints)
        objectiveFunction = self.__minimize_distance_and_time_objective_function
        # perform optimization
        result = minimize(
            objectiveFunction,
            x0=optimization_variables,
            method='SLSQP', 
            constraints=(waypoint_constraint))
        # retrieve data
        optimized_control_points = self.__create_list_of_control_points_per_spline(result.x)
        optimized_scale_factors = self.__create_list_of_scale_factors(result.x)
        self._control_point_list = optimized_control_points
        self._scale_factors = optimized_scale_factors
        
    def __minimize_distance_and_time_objective_function(self,variables):
        control_points_per_spline = self.__create_list_of_control_points_per_spline(variables)
        scale_factors = self.__create_list_of_scale_factors(variables)
        distance = 0
        time = 0
        for i in range(self._number_of_splines):
            control_points = control_points_per_spline[i]
            number_of_intervals = self._number_of_intervals_per_spline[i]
            distance += self.__calculate_total_distance_between_points(control_points)
            time += scale_factors[i]*number_of_intervals
        return distance*time

    def __create_list_of_control_points_per_spline(self,variables):
        number_of_control_points = int((len(variables)-self._number_of_splines)/self._dimension)
        control_points = np.reshape(variables[0:number_of_control_points*self._dimension],(self._dimension,number_of_control_points))
        control_points_per_spline_list = []
        for i in range(self._number_of_splines):
            start_index = self._spline_first_control_point_indices[i]
            if i == self._number_of_splines -1:
                end_index = number_of_control_points
            else:
                end_index = self._spline_first_control_point_indices[i+1]
            control_points_per_spline_list.append(control_points[:,start_index:end_index])
        return control_points_per_spline_list

    def __create_list_of_scale_factors(self,variables):
        scale_factors = variables[ len(variables)-self._number_of_splines, len(variables)]
        return scale_factors

    def __calculate_distances_between_points(self,points):
        number_of_points = np.shape(points)[1]
        first_points = points[:,0:number_of_points-1]
        next_points = points[:,1:number_of_points]
        distances = np.sqrt(np.sum(((next_points - first_points)**2),0))
        return distances

    def __calculate_total_distance_between_points(self, points):
        distances = self.__calculate_distances_between_points(points)
        return np.sum(distances)

    def __get_number_of_waypoints(self, waypoints):
        mu = np.shape(waypoints)[1]
        return mu

    def __get_number_of_splines(self, waypoints):
        beta = self.__get_number_of_waypoints(waypoints)-1
        return beta

    def __get_number_of_intervals_per_spline_array(self, waypoints):
        distances = self.__calculate_distances_between_points(waypoints)
        nu_array = np.ceil(distances / self._interval_spacing_length)
        return nu_array

    def __get_number_of_control_points_per_spline_array(self, waypoints):
        nu_array = self.__get_number_of_intervals_per_spline_array(waypoints)
        n_array = nu_array + self._order
        return n_array

    def __get_spline_initial_control_point_indices_array(self, waypoints):
        num_control_points_per_spline = self.__get_number_of_control_points_per_spline_array(waypoints)
        num_splines = self.__get_number_of_splines(waypoints)
        first_control_point_indices = np.zeros(num_splines)
        index_count = 0
        for i in range(num_splines):
            first_control_point_indices[i] = index_count
            index_count += num_control_points_per_spline[i]
        return first_control_point_indices
            
    def __get_number_of_control_points(self, waypoints):
        n_total = self.__get_number_of_control_points_per_spline_array(waypoints)
        return n_total

    def __create_initial_set_of_control_points(self, waypoints):
        n_array = self.__get_number_of_control_points_per_spline_array(waypoints)
        beta = self.__get_number_of_splines(waypoints)
        control_points = np.array([])
        for spline_number in range(beta):
            start_waypoint = waypoints[:,spline_number]
            end_waypoint = waypoints[:,spline_number+1]
            spline_i_control_points = np.linspace(start_waypoint,end_waypoint,n_array[spline_number]).T
            if spline_number == 0:
                control_points = spline_i_control_points
            else:
                control_points = np.concatenate((spline_i_control_points,control_points),1)
        return control_points

    def __create_initial_scale_factors(self, number_of_splines):
        return np.ones(number_of_splines)

    def __create_waypoint_constraint(self, waypoints, dimension):
        M = get_M_matrix(0, self._order, np.array([]), False)
        T_0 = get_T_vector(self._order,0,0,0,1)
        T_f = get_T_vector(self._order,1,0,0,1)
        num_control_points = self.__get_number_of_control_points(waypoints)
        num_splines = self.__get_number_of_splines(waypoints)
        num_control_points_per_spline = self.__get_number_of_control_points_per_spline_array(waypoints, self._order)
        constraint_matrix = np.zeros((num_splines*2, num_control_points))
        M_T0 = np.dot(M,T_0).flatten()
        M_Tf = np.dot(M,T_f).flatten()
        points_count = 0
        waypoint_constraints = np.zeros((dimension,num_splines*2))
        for i in range(num_splines):
            num_points = num_control_points_per_spline[i]
            constraint_matrix[2*i,points_count:points_count+self._order+1] = M_T0
            points_count += num_points 
            constraint_matrix[2*i+1,points_count-self._order-1:points_count] = M_Tf
            waypoint_constraints[:,2*i] = waypoints[:,i]
            waypoint_constraints[:,2*i+1] = waypoints[:,i+1]
        constraint_matrix = np.kron(np.eye(dimension),constraint_matrix)
        constraint = LinearConstraint(constraint_matrix, lb=waypoint_constraints.flatten(), ub=waypoint_constraints.flatten())
        return constraint

    def plot_trajectory(self, control_point_list, scale_factor_list):

        









        