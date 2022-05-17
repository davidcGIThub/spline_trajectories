"""
This module contains code to generate B-spline trajectories with constraints
over the position of the trajectory and it's derivatives. It may also constrain
the curvature as well as region avoidance.
"""

import numpy as np 
from scipy.optimize import minimize, Bounds, LinearConstraint
from bsplinegenerator.matrix_evaluation import get_M_matrix
from enum import Enum

class ObjectiveType(Enum):
    MINIMIZE_TIME = 1
    MINIMIZE_DISTANCE = 2
    MINIMIZE_TIME_AND_DISTANCE = 3
    MINIMIZE_JERK = 4
    MINIMIZE_SNAP = 5

class ConstrainedTrajectory:
    """
    This module contains code to generate B-spline trajectories through some waypoints
    with constraints over the derivatives of the trajectory, and combined derivative magnitudes.
    It may also constrain the curvature as well as region avoidance.
    """

    def __init__(self, objectiveType, order, dimension, max_interval_distance = 1, control_point_bounds = [-100,100]):
        self._objectiveType = objectiveType
        self._order = order
        self._max_interval_distance = max_interval_distance
        self._control_point_bounds = control_point_bounds
        self._dimension = dimension

    def generate_trajectory(self, waypoints):
        if self._objectiveType == ObjectiveType.MINIMIZE_TIME:
            objectiveFunction = self.__minimize_time_objective_function
        elif self._objectiveType == ObjectiveType.MINIMIZE_DISTANCE:
            objectiveFunction = self.__minimize_distance_objective_function
        elif self._objectiveType == ObjectiveType.MINIMIZE_TIME_AND_DISTANCE:
            objectiveFunction = self.__minimize_distance_and_time_objective_function
        elif self._objectiveType == ObjectiveType.MINIMIZE_JERK:
            objectiveFunction = self.__minimize_jerk_objective_function
        elif self._objectiveType == ObjectiveType.MINIMIZE_SNAP:
            objectiveFunction = self.__minimize_snap_objective_function
        initial_control_points = self.__create_interpolated_points(waypoints)
        number_of_control_points = np.shape(initial_control_points)[1]
        initial_scale_factor = 1
        optimization_variables = np.concatenate((initial_control_points.flatten(),[initial_scale_factor]))
        optimization_variable_lower_bound = optimization_variables*0 + self._control_point_bounds[0]
        optimization_variable_upper_bound = optimization_variables*0 + self._control_point_bounds[1]
        optimization_variable_lower_bound[-1] = 0.0001
        optimization_variable_bounds = Bounds(lb=optimization_variable_lower_bound, ub = optimization_variable_upper_bound)
        waypoint_constraint = self.__compose_waypoint_constraint(waypoints, number_of_control_points)
        result = minimize(
            objectiveFunction,
            x0=optimization_variables,
            bounds = optimization_variable_bounds, 
            constraints=(waypoint_constraint))
        optimized_scale_factor = result.x[-1]
        control_points_optimized = result.x[0:number_of_control_points*self._dimension].reshape(self._dimension,number_of_control_points)
        return control_points_optimized, optimized_scale_factor

    def __minimize_time_objective_function(self,variables):
        number_of_control_points = int((len(variables) - 1)/2)
        scale_factor = variables[-1]
        time = scale_factor * (number_of_control_points - 2*self._order)
        return time

    def __minimize_distance_objective_function(self,variables):
        number_of_control_points = int((len(variables)-1)/2)
        control_points = np.reshape(variables[0:number_of_control_points*2],(2,number_of_control_points))
        distances = self.__get_distances_between_points(control_points)
        # Try minimizing distance between Bezeir control points in the future
        return np.sum(distances)

    def __minimize_distance_and_time_objective_function(self,variables):
        number_of_control_points = int((len(variables)-1)/2)
        control_points = np.reshape(variables[0:number_of_control_points*2],(2,number_of_control_points))
        distances = self.__get_distances_between_points(control_points)
        scale_factor = variables[-1]
        return np.sum(distances)*scale_factor

    def __minimize_jerk_objective_function(self,variables):
        #TODO
        pass

    def __minimize_snap_objective_function(self,variables):
        #TODO
        pass

    def __compose_waypoint_constraint(self, waypoints, number_of_control_points):
        M = get_M_matrix(0, self._order, np.array([]), False)
        Gamma_0 = np.zeros((self._order+1,1))
        Gamma_0[self._order,0] = 1
        Gamma_f = np.ones((self._order+1,1))
        M_Gamma_0 = np.dot(M,Gamma_0)
        M_Gamma_f = np.dot(M,Gamma_f)
        number_of_waypoints = np.shape(waypoints)[1]
        correlation_map = self.__correlate_waypoints_to_control_points(waypoints,number_of_control_points)
        constraint_sub_matrix = np.zeros((number_of_waypoints , number_of_control_points))
        for i in range(number_of_waypoints):
            control_point_index = int(correlation_map[i])
            constraint_sub_matrix[i,control_point_index:control_point_index+self._order+1][:,None] = M_Gamma_0
            if i == (number_of_waypoints-1):
                constraint_sub_matrix[i,control_point_index:control_point_index+self._order+1][:,None] = M_Gamma_f
        constraint_matrix_top = np.concatenate((constraint_sub_matrix,np.zeros((number_of_waypoints , number_of_control_points))),0)
        constraint_matrix_bottom = np.concatenate((np.zeros((number_of_waypoints , number_of_control_points)),constraint_sub_matrix),0)
        constraint_matrix = np.concatenate((constraint_matrix_top,constraint_matrix_bottom),1)
        scale_factor_column = np.zeros((number_of_waypoints*self._dimension,1))
        constraint_matrix  = np.concatenate((constraint_matrix,scale_factor_column),1)
        constraint = LinearConstraint(constraint_matrix, lb=waypoints.flatten(), ub=waypoints.flatten())
        return constraint

    def __get_distances_between_points(self,points):
        number_of_points = np.shape(points)[1]
        first_points = points[:,0:number_of_points-1]
        next_points = points[:,1:number_of_points]
        distances = np.sqrt(np.sum(((next_points - first_points)**2),0))
        return distances

    def __correlate_waypoints_to_control_points(self,waypoints,number_of_control_points):
        distances = self.__get_distances_between_points(waypoints)
        resolution = np.sum(distances) / (number_of_control_points- 1 - self._order)
        distance_intervals = np.concatenate((np.array([0]) , self.__create_array_of_sums_of_previous_elements(distances)))
        number_of_points = np.shape(waypoints)[1]
        correlation_map = np.zeros(int(number_of_points))
        for i in range(number_of_points-1):
            correlation_map[i] = int(distance_intervals[i] / resolution)
        correlation_map[number_of_points-1] = int(number_of_control_points - self._order -1)
        return correlation_map

    def __create_array_of_sums_of_previous_elements(self, array):
        new_array = np.copy(array)
        for i in range(1,len(new_array)):
            new_array[i] = new_array[i] + new_array[i-1]
        return new_array

    def __create_interpolated_points(self, original_points):
        distances = self.__get_distances_between_points(original_points)
        min_interval_length = np.min([np.min(distances),self._max_interval_distance])
        number_of_new_points = int(np.ceil(np.sum(distances)/min_interval_length) + self._order)
        resolution = np.sum(distances) / (number_of_new_points-1)
        distance_intervals = np.concatenate((np.array([0]) , self.__create_array_of_sums_of_previous_elements(distances)))
        new_points = np.zeros((original_points.ndim,number_of_new_points))
        for i in range(number_of_new_points-1):
            current_distance = resolution*i
            point_end_index = np.where(distance_intervals>current_distance)[0].item(0)
            point_1 = original_points[:,point_end_index-1]
            point_2 = original_points[:,point_end_index]
            ratio_along_segment = (current_distance - distance_intervals[point_end_index-1]) /  (distance_intervals[point_end_index] - distance_intervals[point_end_index-1])
            new_point = point_1 + (point_2 - point_1)*ratio_along_segment
            new_points[:,i] = new_point
        new_points[:,number_of_new_points-1] = original_points[:,-1]
        return new_points
