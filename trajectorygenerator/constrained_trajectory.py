"""
This module contains code to generate B-spline trajectories with constraints
over the position of the trajectory and it's derivatives. It may also constrain
the curvature as well as region avoidance.
"""

import numpy as np 
from scipy.optimize import minimize
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

    def __init__(self, objectiveType, order, control_point_resolution = 1):
        self._objectiveType = objectiveType
        self._order = order
        self._control_point_resolution = control_point_resolution # number of intervals per distance unit


    def generate_trajectory(waypoints):
        #TODO
        pass

    def __minimize_distance_objective_function(self,variables):
        number_of_control_points = int((len(variables)-1)/2)
        control_points = np.reshape(variables[0:number_of_control_points*2],(2,number_of_control_points))
        distances = self.__get_distances_between_points(control_points)
        # Try minimizing distance between Bezeir control points in the future
        return np.sum(distances)

    def __minimize_time_objective_function(self,variables):
        number_of_control_points = int((len(variables) - 1)/2)
        scale_factor = variables[-1]
        time = scale_factor * (number_of_control_points - 2*self._order)
        return time

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

    def __compose_waypoint_constraint_matrix(self, waypoints, number_of_control_points):
        M = get_M_matrix(0, self._order, np.array([]), False)
        Gamma_0 = np.zeros((self._order+1,1))
        Gamma_0[self._order,0] = 1
        Gamma_f = np.ones((self._order+1,1))
        M_Gamma_0 = np.dot(M,Gamma_0)
        M_Gamma_f = np.dot(M,Gamma_f)
        number_of_waypoints = np.shape(waypoints)[1]
        correlation_map = self.__correlate_waypoints_to_control_points(waypoints,number_of_control_points,self._order)
        constraint_sub_matrix = np.zeros((number_of_waypoints , number_of_control_points))
        for i in range(number_of_waypoints):
            control_point_index = int(correlation_map[i])
            constraint_sub_matrix[i,control_point_index:control_point_index+self._order+1][:,None] = M_Gamma_0
            if i == (number_of_waypoints-1):
                constraint_sub_matrix[i,control_point_index:control_point_index+self._order+1][:,None] = M_Gamma_f
        constraint_matrix_top = np.concatenate((constraint_sub_matrix,np.zeros((number_of_waypoints , number_of_control_points))),0)
        constraint_matrix_bottom = np.concatenate((np.zeros((number_of_waypoints , number_of_control_points)),constraint_sub_matrix),0)
        constraint_matrix = np.concatenate((constraint_matrix_top,constraint_matrix_bottom),1)
        return constraint_matrix


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
