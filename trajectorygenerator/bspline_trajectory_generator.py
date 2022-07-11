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


class BsplineTrajectory:
    """
    This module contains code to generate B-spline trajectories through some waypoints
    with constraints over the derivatives of the trajectory, and combined derivative magnitudes.
    It may also constrain the curvature as well as region avoidance.
    """

    def __init__(self, dimension, order, max_interval_distance = 1, control_point_bounds = [-100,100]):
        self._max_interval_distance = max_interval_distance
        self._control_point_bounds = control_point_bounds
        self._dimension = dimension
        self._order = order
        self._max_curvature = 10000
        self._waypoints = np.array([])
        self._waypoint_directions = np.array([])

    def generate_trajectory(self, waypoints,waypoint_directions, max_curvature=np.inf):
        objectiveFunction = self.__minimize_distance_and_time_objective_function
        self._waypoints = waypoints
        self._waypoint_directions = waypoint_directions
        # self._max_curvature = max_curvature
        initial_control_points = self.__create_interpolated_points(waypoints)
        number_of_control_points = np.shape(initial_control_points)[1]
        initial_scale_factor = 1
        optimization_variables = np.concatenate((initial_control_points.flatten(),[initial_scale_factor]))
        optimization_variable_lower_bound = optimization_variables*0 + self._control_point_bounds[0]
        optimization_variable_upper_bound = optimization_variables*0 + self._control_point_bounds[1]
        optimization_variable_lower_bound[-1] = 0.0001
        optimization_variable_bounds = Bounds(lb=optimization_variable_lower_bound, ub = optimization_variable_upper_bound)
        waypoint_constraint = self.__create_waypoint_constraint(waypoints, number_of_control_points)
        # waypoint_direction_constraint = self.__create_direction_constraint()
        result = minimize(
            objectiveFunction,
            x0=optimization_variables,
            method='SLSQP',
            bounds = optimization_variable_bounds,
            constraints=(waypoint_constraint))
        optimized_scale_factor = result.x[-1]
        control_points_optimized = result.x[0:number_of_control_points*self._dimension].reshape(self._dimension,number_of_control_points)
        return control_points_optimized, optimized_scale_factor

    def __minimize_distance_and_time_objective_function(self,variables):
        number_of_control_points = int((len(variables)-1)/2)
        control_points = np.reshape(variables[0:number_of_control_points*2],(2,number_of_control_points))
        distances = self.__get_distances_between_points(control_points)
        scale_factor = variables[-1]
        return np.sum(distances) *scale_factor

    def __create_direction_constraint(self):
        def direction_constraint_function(optimization_variables):
            # has problems with waypoint at [0,0]
            number_of_control_points = int((len(optimization_variables) - 1)/2)
            scale_factor = optimization_variables[-1]
            control_points = np.transpose(np.reshape(optimization_variables[0:number_of_control_points*2],(2,number_of_control_points)))
            D_ = self.compose_point_velocity_constraint_matrix(self._order,self._waypoints, number_of_control_points,scale_factor)
            velocities = np.dot(D_,control_points)
            number_of_angles = np.shape(velocities)[0]
            angles = np.zeros(number_of_angles)
            for i in range(number_of_angles):
                angles[i] = np.arctan2(velocities[i,1],velocities[i,0])
            constraints = self._waypoint_directions - angles   
            return constraints
        lower_bound = np.zeros(np.shape(self._waypoints)[1])
        upper_bound = np.zeros(np.shape(self._waypoints)[1])
        direction_constraint = NonlinearConstraint(direction_constraint_function, lb=lower_bound,ub=upper_bound)
        return direction_constraint

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