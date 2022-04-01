  def __cox_de_boor_table_method(self, time):
        """
        This function evaluates the B spline at the given time
        """
        preceding_knot_index = find_preceding_knot_index(time, self._order, self._knot_points)
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
        number_of_control_points = count_number_of_control_points(self._control_points)
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