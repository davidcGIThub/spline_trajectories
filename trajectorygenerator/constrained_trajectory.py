"""
This module contains code to generate B-spline trajectories with constraints
over the position of the trajectory and it's derivatives. It may also constrain
the curvature as well as region avoidance.
"""

from matplotlib import scale
from matplotlib.pyplot import sca
import numpy as np 
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from bsplinegenerator.matrix_evaluation import get_M_matrix
from enum import Enum

class ObjectiveType(Enum):
    MINIMIZE_TIME = 1
    MINIMIZE_DISTANCE = 2
    MINIMIZE_TIME_AND_DISTANCE = 3
    MINIMIZE_VELOCITY = 4
    MINIMIZE_ACCELERATION = 5
    MINIMIZE_JERK = 6
    MINIMIZE_SNAP = 7

class ConstrainedTrajectory:
    """
    This module contains code to generate B-spline trajectories through some waypoints
    with constraints over the derivatives of the trajectory, and combined derivative magnitudes.
    It may also constrain the curvature as well as region avoidance.
    """

    def __init__(self, objectiveType, dimension, max_interval_distance = 1, control_point_bounds = [-100,100]):
        self._objectiveType = objectiveType
        self._max_interval_distance = max_interval_distance
        self._control_point_bounds = control_point_bounds
        self._dimension = dimension
        self._order = 5
        self._max_curvature = 10000

    def generate_trajectory(self, waypoints, max_curvature=np.inf):
        if self._objectiveType == ObjectiveType.MINIMIZE_TIME:
            objectiveFunction = self.__minimize_time_objective_function
        elif self._objectiveType == ObjectiveType.MINIMIZE_DISTANCE:
            objectiveFunction = self.__minimize_distance_objective_function
        elif self._objectiveType == ObjectiveType.MINIMIZE_TIME_AND_DISTANCE:
            objectiveFunction = self.__minimize_distance_and_time_objective_function
        elif self._objectiveType == ObjectiveType.MINIMIZE_VELOCITY:
            objectiveFunction = self.__minimize_velocity_objective_function
        elif self._objectiveType == ObjectiveType.MINIMIZE_ACCELERATION:
            objectiveFunction = self.__minimize_acceleration_objective_function
        elif self._objectiveType == ObjectiveType.MINIMIZE_JERK:
            objectiveFunction = self.__minimize_jerk_objective_function
        elif self._objectiveType == ObjectiveType.MINIMIZE_SNAP:
            objectiveFunction = self.__minimize_snap_objective_function
        self._max_curvature = max_curvature
        initial_control_points = self.__create_interpolated_points(waypoints)
        number_of_control_points = np.shape(initial_control_points)[1]
        print("number_control_points: " , number_of_control_points )
        initial_scale_factor = 1
        optimization_variables = np.concatenate((initial_control_points.flatten(),[initial_scale_factor]))
        optimization_variable_lower_bound = optimization_variables*0 + self._control_point_bounds[0]
        optimization_variable_upper_bound = optimization_variables*0 + self._control_point_bounds[1]
        optimization_variable_lower_bound[-1] = 0.0001
        optimization_variable_bounds = Bounds(lb=optimization_variable_lower_bound, ub = optimization_variable_upper_bound)
        waypoint_constraint = self.__create_waypoint_constraint(waypoints, number_of_control_points)
        result = minimize(
            objectiveFunction,
            x0=optimization_variables,
            method='SLSQP',
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

    def __minimize_velocity_objective_function(self,variables):
        number_of_control_points = int((len(variables)-1)/2)
        control_points = np.reshape(variables[0:number_of_control_points*2],(2,number_of_control_points))
        scale_factor = variables[-1]
        summation = 0
        for i in range(0,number_of_control_points-5):
            p1 = control_points[:,i]
            p2 = control_points[:,i+1]
            p3 = control_points[:,i+2]
            p4 = control_points[:,i+3]
            p5 = control_points[:,i+4]
            p6 = control_points[:,i+5]
            integrands = 1/(181440*scale_factor)*(35*p1**2 + p1*(1051*p2+460*p3-1330*p4 - 260*p5-p6)
                + 10319*p2**2 + 19726*p2*p3 - 5*p2*(6044*p4 + 2689*p5 + 50*p6) + 23624*p3**2
                - 35884*p3*p4 - 43520*p3*p5 - 1330*p3*p6 + 23624*p4**2 + 24326*p4*p5 + 460*p4*p6 + 24329*p5**2 
                + 1751*p5*p6 + 35*p6**2)
            summation += np.sum(integrands)
        return summation

    def __minimize_acceleration_objective_function(self,variables):
        number_of_control_points = int((len(variables)-1)/2)
        control_points = np.reshape(variables[0:number_of_control_points*2],(2,number_of_control_points))
        scale_factor = variables[-1]
        summation = 0
        for i in range(0,number_of_control_points-5):
            p1 = control_points[:,i]
            p2 = control_points[:,i+1]
            p3 = control_points[:,i+2]
            p4 = control_points[:,i+3]
            p5 = control_points[:,i+4]
            p6 = control_points[:,i+5]
            integrands = 1/(2520*scale_factor**3)*(10*p1**2 + p1*(89*p2 - 178*p3 + 10*p4 + 68*p5 + p6)
               + 376*p2**2 + p2*(-958*p3 - 638*p4 + 1277*p5 + 58*p6) +
               916*p3**2 - 68*p3*p4 - 538*p3*p5 + 10*p3*p6 + 916*p4**2 -
               2738*p4*p5 - 178*p4*p6 + 2266*p5**2 + 289*p5*p6 + 10*p6**2)
            summation += np.sum(integrands)
        return summation

    def __minimize_jerk_objective_function(self,variables):
        #TODO
        pass

    def __minimize_snap_objective_function(self,variables):
        number_of_control_points = int((len(variables)-1)/2)
        control_points = np.reshape(variables[0:number_of_control_points*2],(2,number_of_control_points))
        scale_factor = variables[-1]
        summation = 0
        for i in range(0,number_of_control_points-5):
            p1 = control_points[:,i]
            p2 = control_points[:,i+1]
            p3 = control_points[:,i+2]
            p4 = control_points[:,i+3]
            p5 = control_points[:,i+4]
            p6 = control_points[:,i+5]
            integrands = ((p1-4*p2+6*p3-4*p4+p5)**3 - (p2-4*p3+6*p4+6*p5+p6)**3)/(3*(scale_factor**7)*(p1-5*(p2-2*p3+2*p4+p5)-p6))
            summation += np.sum(integrands)
        return summation

    def compose_point_velocity_constraint_matrix(self, order,waypoints, number_of_control_points,alpha):
        M = get_M_matrix(0, self._order, np.array([]), False)
        L_0 = np.zeros((order+1,1))
        L_0[order-1,0] = 1/alpha
        L_f = np.zeros((order+1,1))
        for i in range(order):
            L_f[i] = (order-i)/(alpha)
        # L_f[i] = np.math.factorial(order-i)/(alpha**rth_derivative * np.math.factorial(order-rth_derivative-i) )
        M_L_0 = np.dot(M,L_0)
        M_L_f = np.dot(M,L_f)
        number_of_waypoints = np.shape(waypoints)[1]
        correlation_map = self.__correlate_waypoints_to_control_points(waypoints,number_of_control_points)
        constraint_sub_matrix = np.zeros((number_of_waypoints , number_of_control_points))
        for i in range(number_of_waypoints):
            control_point_index = int(correlation_map[i])
            constraint_sub_matrix[i,control_point_index:control_point_index+order+1][:,None] = M_L_0
            if i == (number_of_waypoints-1):
                constraint_sub_matrix[i,control_point_index:control_point_index+order+1][:,None] = M_L_f
        return constraint_sub_matrix

    def __create_waypoint_constraint(self, waypoints, number_of_control_points):
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
        print("number_of_waypoints: ", number_of_waypoints)
        print("self._dimension: ", self._dimension)
        scale_factor_column = np.zeros((number_of_waypoints*self._dimension,1))
        print(np.shape(constraint_matrix))
        print(np.shape(scale_factor_column))
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


    # def __create_curvature_constraint(self, number_of_control_points):
    #     def curvature_constraint_function(variables):
    #         number_of_control_points = int((len(variables)-1)/2)
    #         control_points = np.reshape(variables[0:number_of_control_points*2],(2,number_of_control_points))
    #         constraints = np.zeros(3*(number_of_control_points-2))
    #         print("control_points: " , control_points)
    #         for i in range(0,number_of_control_points-5):
    #             point1 = control_points[:,i]
    #             point2 = control_points[:,i+1]
    #             point3 = control_points[:,i+2]
    #             point4 = control_points[:,i+3]
    #             point5 = control_points[:,i+4]
    #             point6 = control_points[:,i+5]
    #             vec1 = point1 - point2
    #             vec2 = point3 - point2
    #             vec
    #             vec1_mag = np.linalg.norm(vec1)
    #             vec2_mag = np.linalg.norm(vec2)
    #             cos_theta = np.dot(vec1,vec2)/(vec1_mag*vec2_mag)
    #             constraints[3*i] = 4*(1-cos_theta**2)/vec1_mag**2 - self._max_curvature**2
    #             constraints[3*i + 1] = 4*(1-cos_theta**2)/vec2_mag**2 - self._max_curvature**2
    #             constraints[3*i + 2] = np.dot(vec1,vec2)
    #         print("constraints: " , constraints)
    #         return constraints
    #     curvature_constraint_lower_bound = np.zeros((number_of_control_points-2)*3) - np.inf
    #     curvature_constraint_upper_bound = np.zeros((number_of_control_points-2)*3)
    #     curvature_constraint = NonlinearConstraint(curvature_constraint_function, lb=curvature_constraint_lower_bound, ub=curvature_constraint_upper_bound)
    #     return curvature_constraint