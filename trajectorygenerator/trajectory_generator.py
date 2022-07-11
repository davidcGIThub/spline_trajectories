"""
This module contains code to generate B-spline trajectories with constraints
over the position of the trajectory and it's derivatives. It may also constrain
the curvature as well as region avoidance. ** currently works for dimension 2-3, and order***
"""
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from bsplinegenerator.matrix_evaluation import get_M_matrix, get_T_vector, get_T_derivative_vector

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
        self._control_point_vector = np.array([])
        
    def generate_trajectory(self, waypoints, velocity_waypoints, max_velocity,max_acceleration, max_turn_rate):
        # create initial conditions
        max_curvature = max_turn_rate/max_velocity
        self._dimension = np.shape(waypoints)[0]
        self._number_of_splines = self.__get_number_of_splines(waypoints)
        self._number_of_intervals_per_spline = self.__get_number_of_intervals_per_spline_array(waypoints)
        num_splines = self.__get_number_of_splines(waypoints)
        num_control_points = self.__get_number_of_control_points(waypoints)
        initial_control_points = self.__create_initial_set_of_control_points(waypoints)
        initial_scale_factors = self.__create_initial_scale_factors(num_splines)
        optimization_variables = np.concatenate((initial_control_points.flatten(),initial_scale_factors))
        # define constraints and objective function and constraints
        waypoint_constraint = self.__create_waypoint_constraint(waypoints, self._dimension)
        objective_variable_bounds = self.__create_objective_variable_bounds(waypoints)
        objectiveFunction = self.__minimize_distance_and_time_objective_function
        # g1_continuity_constraint = self.__create_G1_continuity_constraint()
        g2_continuity_constraint = self.__create_G2_continuity_constraint()
        velocity_waypoint_constraint = self.__create_velocity_waypoint_constraints(velocity_waypoints)
        max_velocity_constraint = self.__create_max_velocity_constraint(max_velocity, num_control_points)
        max_acceleration_constraint = self.__create_acceleration_constraint(max_acceleration, num_control_points)
        curvature_constraint = self.__create_curvature_constraint(max_curvature)
        minimize_options = {'disp': True}#, 'maxiter': self.maxiter, 'ftol': tol}
        # perform optimizationd
        result = minimize(
            objectiveFunction,
            x0=optimization_variables,
            method='SLSQP', 
            # method = 'trust-constr',
            constraints=(waypoint_constraint, velocity_waypoint_constraint,max_velocity_constraint,
                g2_continuity_constraint,max_acceleration_constraint,curvature_constraint),
            bounds=objective_variable_bounds, 
            options = minimize_options)
        # retrieve data
        optimized_control_points = self.__create_list_of_control_points(result.x)
        optimized_scale_factors = self.__create_list_of_scale_factors(result.x)
        self._control_point_list = optimized_control_points
        self._scale_factor_list = optimized_scale_factors
        return self._control_point_list, self._scale_factor_list

    def get_initial_control_points(self, waypoints):
        return self.__create_initial_set_of_control_points(waypoints)
        
    def __minimize_distance_and_time_objective_function(self,variables):
        control_points_per_spline = self.__create_list_of_control_points(variables)
        scale_factors = self.__create_list_of_scale_factors(variables)
        distance = 0
        time = 0
        objective = 0
        for i in range(self._number_of_splines):
            number_of_intervals = self._number_of_intervals_per_spline[i]
            time = scale_factors[i]*number_of_intervals
            control_points = control_points_per_spline[i]
            distance = self.__calculate_total_distance_between_points(control_points)
            objective += distance + time
        return objective

    def __create_list_of_control_points(self,variables):
        number_of_control_points = int((len(variables)-self._number_of_splines)/self._dimension)
        control_points = np.reshape(variables[0:number_of_control_points*self._dimension],(self._dimension,number_of_control_points))
        control_points_list = []
        control_point_count = 0
        for i in range(self._number_of_splines):
            control_points_per_spline = self._number_of_intervals_per_spline[i] + self._order
            start_index = control_point_count
            if i == self._number_of_splines -1:
                end_index = number_of_control_points
            else:
                end_index = control_points_per_spline + control_point_count
            control_points_list.append(control_points[:,start_index:end_index])
            control_point_count += control_points_per_spline
        return control_points_list

    def __create_list_of_scale_factors(self,variables):
        scale_factors = variables[ len(variables)-self._number_of_splines: len(variables)].tolist()
        return scale_factors

    def __calculate_distances_between_points(self,points):
        number_of_points = np.shape(points)[1]
        first_points = points[:,0:number_of_points-1]
        next_points = points[:,1:number_of_points]
        distances = np.sqrt(np.sum(((next_points - first_points)**2),0))
        return distances

    def __calculate_total_distance_between_points(self, points):
        distance = np.sum(self.__calculate_distances_between_points(points))
        return distance

    def __get_number_of_waypoints(self, waypoints):
        mu = np.shape(waypoints)[1]
        return mu

    def __get_number_of_splines(self, waypoints):
        beta = self.__get_number_of_waypoints(waypoints)-1
        return beta

    def __get_number_of_intervals_per_spline_array(self, waypoints):
        distances = self.__calculate_distances_between_points(waypoints)
        nu_array = np.ceil(distances / self._interval_spacing_length).astype(int)
        return nu_array

    def __get_number_of_control_points_per_spline_array(self, waypoints):
        nu_array = self.__get_number_of_intervals_per_spline_array(waypoints)
        n_array = nu_array + self._order
        return n_array
            
    def __get_number_of_control_points(self, waypoints):
        n_total = np.sum(self.__get_number_of_control_points_per_spline_array(waypoints))
        return int(n_total)

    def __create_initial_set_of_control_points(self, waypoints):
        n_array = self.__get_number_of_control_points_per_spline_array(waypoints)
        beta = self.__get_number_of_splines(waypoints)
        control_points = np.array([])
        for spline_number in range(beta):
            start_waypoint = waypoints[:,spline_number]
            end_waypoint = waypoints[:,spline_number+1]
            spline_i_control_points = np.linspace(start_waypoint,end_waypoint,int(n_array[spline_number])).T
            if spline_number == 0:
                control_points = spline_i_control_points
            else:
                control_points = np.concatenate((spline_i_control_points,control_points),1)
        return control_points

    def __create_initial_scale_factors(self, number_of_splines):
        return np.ones(number_of_splines)

    def __create_waypoint_constraint(self, waypoints, dimension):
        M = get_M_matrix(0, self._order, np.array([]), False)
        T_0 = get_T_vector(self._order,0,0,1)
        T_f = get_T_vector(self._order,1,0,1)
        num_control_points = self.__get_number_of_control_points(waypoints)
        num_splines = self.__get_number_of_splines(waypoints)
        num_control_points_per_spline_array = self.__get_number_of_control_points_per_spline_array(waypoints)
        constraint_matrix = np.zeros((num_splines*2, num_control_points))
        M_T0 = np.dot(M,T_0).flatten()
        M_Tf = np.dot(M,T_f).flatten()
        points_count = 0
        waypoint_constraints = np.zeros((dimension,num_splines*2))
        for i in range(num_splines):
            num_points = num_control_points_per_spline_array[i]
            constraint_matrix[2*i,points_count:points_count+self._order+1] = M_T0
            points_count += num_points 
            constraint_matrix[2*i+1,points_count-self._order-1:points_count] = M_Tf
            waypoint_constraints[:,2*i] = waypoints[:,i]
            waypoint_constraints[:,2*i+1] = waypoints[:,i+1]
        constraint_matrix = np.kron(np.eye(dimension),constraint_matrix)
        constraint_matrix = np.concatenate((constraint_matrix, np.zeros((num_splines*2*self._dimension,num_splines))),1)
        constraint = LinearConstraint(constraint_matrix, lb=waypoint_constraints.flatten(), ub=waypoint_constraints.flatten())
        return constraint

    def __create_max_velocity_constraint(self, max_velocity, total_num_control_points):
        def max_velocity_constraint_function(variables):
            control_points_per_spline = self.__create_list_of_control_points(variables)
            scale_factors = self.__create_list_of_scale_factors(variables)
            max_velocity_constraints = np.zeros(total_num_control_points - self._number_of_splines)
            count = 0
            for i in range(self._number_of_splines):
                scale_factor = scale_factors[i]
                control_points = control_points_per_spline[i]
                num_control_points = np.shape(control_points)[1]
                points_previous = control_points[: , 0:num_control_points-1]
                points_next = control_points[: , 1:num_control_points]
                velocities = (points_next - points_previous) / scale_factor
                velocities_magnitudes_squared = np.sum(velocities**2,0)
                max_velocity_constraints[count:count+num_control_points-1] = velocities_magnitudes_squared - max_velocity**2
                count += num_control_points-1
            return max_velocity_constraints
        lower_bound = np.zeros(total_num_control_points - self._number_of_splines) - np.inf
        upper_bound = np.zeros(total_num_control_points - self._number_of_splines)
        max_velocity_constraint = NonlinearConstraint(max_velocity_constraint_function, lb=lower_bound,ub=upper_bound)
        return max_velocity_constraint

    def __create_acceleration_constraint(self, max_acceleration, total_num_control_points):
        def max_acceleration_constraint_function(variables):
            control_points_per_spline = self.__create_list_of_control_points(variables)
            scale_factors = self.__create_list_of_scale_factors(variables)
            max_acceleration_constraints = np.zeros(total_num_control_points - self._number_of_splines*2)
            count = 0
            for i in range(self._number_of_splines):
                scale_factor = scale_factors[i]
                control_points = control_points_per_spline[i]
                num_control_points = np.shape(control_points)[1]
                points_previous = control_points[: , 0:num_control_points-1]
                points_next = control_points[: , 1:num_control_points]
                velocities = (points_next - points_previous) / scale_factor
                velocities_previous = velocities[:,0:num_control_points-2]
                velocities_next = velocities[:,1:num_control_points-1]
                accelerations = (velocities_next - velocities_previous) / scale_factor
                acceleration_magnitudes_squared = np.sum(accelerations**2,0)
                max_acceleration_constraints[count:count+num_control_points-2] = acceleration_magnitudes_squared - max_acceleration**2
                count += num_control_points-2
            return max_acceleration_constraints
        lower_bound = np.zeros(total_num_control_points - self._number_of_splines*2) - np.inf
        upper_bound = np.zeros(total_num_control_points - self._number_of_splines*2)
        max_acceleration_constraint = NonlinearConstraint(max_acceleration_constraint_function, lb=lower_bound,ub=upper_bound)
        return max_acceleration_constraint

    def __create_velocity_waypoint_constraints(self, velocity_waypoints):
        def velocity_waypoint_constraint_function(variables):
            M = get_M_matrix(0, self._order, np.array([]), False)
            control_points_per_spline = self.__create_list_of_control_points(variables)
            scale_factors = self.__create_list_of_scale_factors(variables)
            dim = self._dimension
            velocity_waypoint_constraints = np.zeros(self._number_of_splines*2*dim)
            for i in range(self._number_of_splines):
                scale_factor = scale_factors[i]
                P_0 = control_points_per_spline[i][:,0:self._order+1]
                P_f = control_points_per_spline[i][:,-(self._order+1):]
                T_0 = get_T_derivative_vector(self._order,0,0,1,scale_factor)
                T_f = get_T_derivative_vector(self._order,scale_factor,0,1,scale_factor)
                velocity_waypoint_constraints[i*2*dim:i*2*dim+dim] = np.dot(P_0,np.dot(M,T_0)).flatten() - velocity_waypoints[:,i]
                velocity_waypoint_constraints[i*2*dim+dim:(i+1)*2*dim] = np.dot(P_f,np.dot(M,T_f)).flatten() - velocity_waypoints[:,i+1]
            return velocity_waypoint_constraints
        lower_bound = np.zeros(self._number_of_splines*2*self._dimension)
        upper_bound = np.zeros(self._number_of_splines*2*self._dimension)
        velocity_waypoint_constraint = NonlinearConstraint(velocity_waypoint_constraint_function, lb=lower_bound,ub=upper_bound)
        return velocity_waypoint_constraint

    def __create_objective_variable_bounds(self, waypoints):
        num_control_points = self.__get_number_of_control_points(waypoints)
        num_scale_factors = self.__get_number_of_splines(waypoints)
        num_variables = num_control_points*self._dimension + num_scale_factors
        lower_bounds = np.zeros(num_variables) - np.inf
        upper_bounds = np.zeros(num_variables) + np.inf
        lower_bounds[num_variables-num_scale_factors:num_variables] = 0.0001
        return Bounds(lb=lower_bounds, ub = upper_bounds)

    def __create_curvature_constraint(self, max_curvature):
        M = get_M_matrix(0, self._order, np.array([]), False)
        total_number_of_intervals = np.sum(self._number_of_intervals_per_spline)
        def max_curvature_objective_function(u_parameter):
            u = u_parameter[0]
            alpha = 1
            dT = get_T_derivative_vector(self._order,u,0,1,alpha)
            d2T = get_T_derivative_vector(self._order,u,0,2,alpha)
            vel_vec = np.dot(self._control_point_vector,np.dot(M,dT)).flatten()
            accel_vec = np.dot(self._control_point_vector,np.dot(M,d2T)).flatten()
            curvature = np.linalg.norm(np.cross(vel_vec,accel_vec))/ np.linalg.norm(vel_vec)**3
            return -curvature
        u_parameter_bounds = Bounds(lb=0, ub = 1.0)
        def curvature_constraint_function(variables):
            control_points_per_spline = self.__create_list_of_control_points(variables)
            curvature_constraints = np.zeros(total_number_of_intervals)
            interval_count = 0
            for i in range(self._number_of_splines):
                control_points = control_points_per_spline[i]
                num_intervals = self._number_of_intervals_per_spline[i]
                for j in range(num_intervals):
                    self._control_point_vector = control_points[:,j:j+self._order+1]
                    t0_1 = np.array([0])
                    t0_2 = np.array([0.5])
                    t0_3 = np.array([1])
                    result_1 = minimize(max_curvature_objective_function, x0=t0_1, method='SLSQP', bounds=u_parameter_bounds)
                    result_2 = minimize(max_curvature_objective_function, x0=t0_2, method='SLSQP', bounds=u_parameter_bounds)
                    result_3 = minimize(max_curvature_objective_function, x0=t0_3, method='SLSQP', bounds=u_parameter_bounds)
                    curvature_1 = -max_curvature_objective_function(np.array([result_1.x]))
                    curvature_2 = -max_curvature_objective_function(np.array([result_2.x]))
                    curvature_3 = -max_curvature_objective_function(np.array([result_3.x]))
                    greatest_curvature = np.max((curvature_1,curvature_2,curvature_3))
                    curvature_constraints[interval_count] = greatest_curvature - max_curvature 
                    interval_count += 1
            return curvature_constraints
        lower_bound = -np.inf
        upper_bound = 0
        curvature_constraint = NonlinearConstraint(curvature_constraint_function, lb=lower_bound,ub=upper_bound)
        return curvature_constraint

    def __create_G1_continuity_constraint(self):
        def g1_continuity_constraint_function(variables):
            M = get_M_matrix(0, self._order, np.array([]), False)
            control_points_per_spline = self.__create_list_of_control_points(variables)
            scale_factors = self.__create_list_of_scale_factors(variables)
            dim = self._dimension
            g1_constraints = np.zeros((self._number_of_splines-1)*dim)
            for i in range(self._number_of_splines-1):
                scale_factor_1 = scale_factors[i]
                scale_factor_2 = scale_factors[i+1]
                P_1 = control_points_per_spline[i][:,-(self._order+1):]
                P_2 = control_points_per_spline[i+1][:,0:self._order+1]
                T_f1 = get_T_derivative_vector(self._order,scale_factor_1,0,1,scale_factor_1)
                T_02 = get_T_derivative_vector(self._order,0,0,1,scale_factor_2)
                g1_constraints[i*dim:i*dim+dim] = (np.dot(P_1 , np.dot(M,T_f1)) - np.dot(P_2,np.dot(M,T_02))).flatten()
            # print("g1_constraints: " , g1_constraints)
            return g1_constraints
        lower_bound = np.zeros((self._number_of_splines-1)*self._dimension)
        upper_bound = np.zeros((self._number_of_splines-1)*self._dimension)
        g1_continuity_constraint = NonlinearConstraint(g1_continuity_constraint_function, lb=lower_bound,ub=upper_bound)
        return g1_continuity_constraint

    def __create_G2_continuity_constraint(self):
        def g2_continuity_constraint_function(variables):
            M = get_M_matrix(0, self._order, np.array([]), False)
            control_points_per_spline = self.__create_list_of_control_points(variables)
            scale_factors = self.__create_list_of_scale_factors(variables)
            dim = self._dimension
            g2_constraints = np.zeros((self._number_of_splines-1)*dim)
            for i in range(self._number_of_splines-1):
                scale_factor_1 = scale_factors[i]
                scale_factor_2 = scale_factors[i+1]
                P_1 = control_points_per_spline[i][:,-(self._order+1):]
                P_2 = control_points_per_spline[i+1][:,0:self._order+1]
                T_f1 = get_T_derivative_vector(self._order,scale_factor_1,0,2,scale_factor_1)
                T_02 = get_T_derivative_vector(self._order,0,0,2,scale_factor_2)
                g2_constraints[i*dim:i*dim+dim] = (np.dot(P_1 , np.dot(M,T_f1)) - np.dot(P_2,np.dot(M,T_02))).flatten()
            # print("g2_constraints: " , g2_constraints)
            return g2_constraints
        lower_bound = np.zeros((self._number_of_splines-1)*self._dimension)
        upper_bound = np.zeros((self._number_of_splines-1)*self._dimension)
        g2_continuity_constraint = NonlinearConstraint(g2_continuity_constraint_function, lb=lower_bound,ub=upper_bound)
        return g2_continuity_constraint
  

## TODO - make better initial conditions, straight line of control points are optimized