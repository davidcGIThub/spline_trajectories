import numpy as np 
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from bsplinegenerator.matrix_evaluation import get_M_matrix, get_T_derivative_vector, get_T_vector
from bsplinegenerator.bsplines import BsplineEvaluation

class PathGenerator:

    def __init__(self, order, dimension, max_interval_distance = 3, control_point_bounds = [-100,100]):
        self._max_interval_distance = max_interval_distance
        self._control_point_bounds = control_point_bounds
        self._dimension = dimension
        self._order = order
        self._max_curvature = np.inf

    def generate_trajectory(self, waypoints, directions, max_curvature=np.inf):
        objectiveFunction = self.__minimize_distance_and_time_objective_function
        self._max_curvature = max_curvature
        initial_control_points = self.__create_interpolated_points(waypoints)
        number_of_control_points = np.shape(initial_control_points)[1]
        initial_scale_factor = 1
        optimization_variables = np.concatenate((initial_control_points.flatten(),[initial_scale_factor]))
        optimization_variable_lower_bound = optimization_variables*0 + self._control_point_bounds[0]
        optimization_variable_upper_bound = optimization_variables*0 + self._control_point_bounds[1]
        optimization_variable_lower_bound[-1] = 0.0001
        optimization_variable_bounds = Bounds(lb=optimization_variable_lower_bound, ub = optimization_variable_upper_bound)
        waypoint_constraint = self.__create_waypoint_constraint(waypoints, number_of_control_points)
        curvature_constraint = self.__create_curvature_constraint(number_of_control_points)
        direction_constraint = self.__create_direction_constraint(waypoints,directions)
        result = minimize(
            objectiveFunction,
            x0=optimization_variables,
            method='SLSQP',
            bounds = optimization_variable_bounds,
            constraints=(waypoint_constraint,direction_constraint,curvature_constraint))
        optimized_scale_factor = result.x[-1]
        control_points_optimized = result.x[0:number_of_control_points*self._dimension].reshape(self._dimension,number_of_control_points)
        return control_points_optimized, optimized_scale_factor

    def __minimize_distance_and_time_objective_function(self,variables):
        number_of_control_points = int((len(variables)-1)/2)
        control_points = np.reshape(variables[0:number_of_control_points*2],(2,number_of_control_points))
        distances = self.__get_distances_between_points(control_points)
        scale_factor = variables[-1]
        return np.sum(distances)*scale_factor

    def __create_curvature_constraint(self,num_control_points):
        acceptable_angle = self.__get_acceptable_angle(self._order)
        def curvature_constraint_function(variables):
            control_points = np.reshape(variables[0:num_control_points*2],(2,num_control_points))
            scale_factor = variables[-1]
            # get max accelerations
            prev_control_points = control_points[:,0:num_control_points-1]
            current_control_points = control_points[:,1:num_control_points]
            control_point_vectors = (current_control_points - prev_control_points)
            prev_vectors = control_point_vectors[:,0:num_control_points-2]
            current_vectors = control_point_vectors[:,1:num_control_points-1]
            max_accelerations = (current_vectors-prev_vectors)/scale_factor**2
            max_accelerations_mag = np.linalg.norm(max_accelerations,2,0)
            # get min velocities
            bcurve = BsplineEvaluation(control_points,self._order,0,scale_factor,False)
            spline_points, time_data = bcurve.get_spline_data(num_control_points)
            prev_spline_points = spline_points[:,0:num_control_points-1]
            current_spline_points = spline_points[:,1:num_control_points]
            min_velocities = (current_spline_points - prev_spline_points)/scale_factor
            min_velocities_mag = np.linalg.norm(min_velocities,2,0)
            # get curvature bounds
            prev_min_velocities_mag = min_velocities_mag[0:num_control_points-2]
            curr_min_velocities_mag = min_velocities_mag[1:num_control_points-1]
            min_velocities_mag = np.min((prev_min_velocities_mag,curr_min_velocities_mag),0)
            curvatures = max_accelerations_mag/min_velocities_mag**2
            curvatures[np.isnan(curvatures)] = 0
            curvature_constraints = curvatures - self._max_curvature
            # get angles between control points using law of cosines
            prior_control_points = control_points[:,0:num_control_points-2]
            next_control_points = control_points[:,2:num_control_points]
            distances = np.linalg.norm(control_point_vectors,2,0)
            prev_distances = distances[0:num_control_points-2]
            current_distances = distances[1:num_control_points-1]
            staggered_distances = np.linalg.norm(next_control_points-prior_control_points,2,0)
            numerator = current_distances**2 + prev_distances**2 - staggered_distances**2
            denominator = 2*current_distances*staggered_distances
            angles = np.arccos(numerator/denominator)
            angles[np.isnan(angles)] = 0
            angle_constraints = angles - acceptable_angle
            #concatenate
            constraints = np.concatenate((curvature_constraints,angle_constraints))
            print("constraints: " , constraints)
            return constraints
        lower_bounds = np.zeros((num_control_points-2)*2) - np.inf
        upper_bounds = np.zeros((num_control_points-2)*2)
        curvature_constraints = NonlinearConstraint(curvature_constraint_function, lb=lower_bounds,ub=upper_bounds)
        return curvature_constraints

    def __get_acceptable_angle(self,order):
        if order == 2:
            angle = np.pi/2 
        elif order == 3:
            angle = np.pi*3/4 
        elif order == 4:
            angle = np.pi*5/6 
        elif order == 5:
            angle = np.pi*7/8 
        return angle

    def __create_direction_constraint(self,waypoints,waypoint_directions):
        def direction_constraint_function(optimization_variables):
            # has problems with waypoint at [0,0]
            number_of_control_points = int((len(optimization_variables) - 1)/2)
            scale_factor = optimization_variables[-1]
            control_points = np.transpose(np.reshape(optimization_variables[0:number_of_control_points*2],(2,number_of_control_points)))
            D_ = self.compose_point_velocity_constraint_matrix(self._order, waypoints, number_of_control_points, scale_factor)
            velocities = np.dot(D_,control_points)
            number_of_angles = np.shape(velocities)[0]
            angles = np.zeros(number_of_angles)
            for i in range(number_of_angles):
                angles[i] = np.arctan2(velocities[i,1],velocities[i,0])
            constraints = waypoint_directions - angles   
            return constraints
        lower_bound = 0
        upper_bound = 0
        direction_constraint = NonlinearConstraint(direction_constraint_function , lb= lower_bound, ub=upper_bound)
        return direction_constraint

    def compose_point_velocity_constraint_matrix(self,order,waypoints, number_of_control_points,alpha):
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

    # def __create_curvature_constraint(self,number_of_control_points):
    #     number_of_intervals = number_of_control_points - 2
    #     def curvature_constraint_function(variables):
    #         control_points = np.reshape(variables[0:number_of_control_points*2],(2,number_of_control_points))
    #         curvature_constraints = np.zeros(number_of_intervals)
    #         for i in range(number_of_control_points-2):
    #             p0 = control_points[:,i]
    #             p1 = control_points[:,i+1]
    #             p2 = control_points[:,i+2]
    #             leg_start = (p1-p0)/2
    #             leg_middle = p1-(p0+p2)/2
    #             leg_end = (p1-p2)/2
    #             A = np.linalg.norm(np.cross(leg_start,leg_end))/2
    #             if np.dot(leg_start,leg_middle) <= 0:
    #                 if A == 0:
    #                     c_max = 0
    #                 else:
    #                     c_max = A/np.linalg.norm(leg_start)**3
    #             elif np.dot(leg_middle,leg_end) <= 0:
    #                 if A == 0:
    #                     c_max = 0
    #                 else:
    #                     c_max = A/np.linalg.norm(leg_end)**3
    #             else:
    #                 if A == 0:
    #                     c_max = np.inf
    #                 else:
    #                     c_max = np.linalg.norm(leg_middle)**3 / A**2
    #             curvature_constraints[i] = c_max - self._max_curvature
    #         return curvature_constraints
    #     lower_bounds = np.zeros(number_of_intervals) - np.inf
    #     upper_bounds = np.zeros(number_of_intervals)
    #     curvature_constraint = NonlinearConstraint(curvature_constraint_function, lb=lower_bounds,ub=upper_bounds)
    #     return curvature_constraint
