class IndicatorConstraints:

    def __init__(self, binv, A, b, sense, indicval):
        self.binv = binv  # index of binary variable
        self.A = A  # CPLEX: lin_expr,   left hand side coefficient row for indicator constraint
        self.b = b  # right hand side for indicator constraint
        self.sense = sense  # sense of the indicator constraint can be 'L', 'E', 'G' (lower-equal, equal, greater-equal)
        self.indicval = indicval  # value the binary variable takes when constraint is fulfilled
