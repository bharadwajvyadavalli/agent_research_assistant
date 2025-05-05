"""
Calculator tool for the agent research assistant.

This module provides mathematical computation capabilities, including:
- Basic arithmetic
- Statistical analysis
- Matrix operations
- Numerical methods
- Symbolic mathematics (optional)
- Unit conversion
"""

import math
import statistics
import time
import uuid
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
import numpy as np
from datetime import datetime

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from utils.logging_utils import AgentLogger


class CalculationResult:
    """Class representing the result of a calculation."""
    
    def __init__(
        self,
        expression: str,
        result: Any,
        result_type: str,
        method: str,
        execution_time: float,
        steps: Optional[List[str]] = None,
        error: Optional[str] = None
    ):
        """
        Initialize a new calculation result.
        
        Args:
            expression: The original expression or operation description
            result: The result of the calculation
            result_type: Type of the result (numeric, array, matrix, etc.)
            method: Method used for calculation
            execution_time: Time taken to execute the calculation (seconds)
            steps: Optional list of intermediate calculation steps
            error: Error message if calculation failed
        """
        self.id = f"calc_{uuid.uuid4().hex[:8]}"
        self.timestamp = datetime.now().isoformat()
        self.expression = expression
        self.result = result
        self.result_type = result_type
        self.method = method
        self.execution_time = execution_time
        self.steps = steps or []
        self.error = error
        self.success = error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the calculation result to a dictionary."""
        # Convert result to a serializable format if needed
        serialized_result = self.result
        if isinstance(self.result, np.ndarray):
            serialized_result = self.result.tolist()
        
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "expression": self.expression,
            "result": serialized_result,
            "result_type": self.result_type,
            "method": self.method,
            "execution_time": self.execution_time,
            "steps": self.steps,
            "error": self.error,
            "success": self.success
        }
    
    def __str__(self) -> str:
        """String representation of the calculation result."""
        if not self.success:
            return f"Error: {self.error}"
        
        if self.result_type == "array" or self.result_type == "matrix":
            return f"Result: {type(self.result).__name__} of shape {np.array(self.result).shape}"
        
        return f"Result: {self.result}"


class CalculationType(Enum):
    """Enumeration of calculation types."""
    ARITHMETIC = "arithmetic"
    STATISTICAL = "statistical"
    MATRIX = "matrix"
    NUMERICAL = "numerical"
    SYMBOLIC = "symbolic"
    UNIT_CONVERSION = "unit_conversion"


class CalculatorTool:
    """
    Tool for performing various mathematical calculations.
    """
    
    def __init__(self, logger: Optional[AgentLogger] = None):
        """
        Initialize the calculator tool.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or AgentLogger(agent_id="calculator_tool")
        self.history: List[CalculationResult] = []
        
        # Set up numpy print options
        np.set_printoptions(precision=8, suppress=True)
    
    def evaluate_expression(self, expression: str) -> CalculationResult:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: The expression to evaluate
            
        Returns:
            CalculationResult object with the result
        """
        start_time = time.time()
        steps = []
        
        try:
            # Remove any potentially unsafe code
            sanitized_expr = self._sanitize_expression(expression)
            steps.append(f"Sanitized expression: {sanitized_expr}")
            
            # Evaluate the expression
            # Note: eval is used in a controlled way here with sanitized input
            local_vars = {
                "math": math,
                "np": np,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "exp": math.exp,
                "log": math.log,
                "log10": math.log10,
                "sqrt": math.sqrt,
                "pi": math.pi,
                "e": math.e
            }
            result = eval(sanitized_expr, {"__builtins__": {}}, local_vars)
            steps.append(f"Evaluated expression using Python's eval function")
            
            # Determine result type
            result_type = self._determine_result_type(result)
            
            # Create and return the result
            calc_result = CalculationResult(
                expression=expression,
                result=result,
                result_type=result_type,
                method="python_eval",
                execution_time=time.time() - start_time,
                steps=steps
            )
            
            self.history.append(calc_result)
            self.logger.info(f"Evaluated expression: {expression}")
            return calc_result
        
        except Exception as e:
            # Handle any errors
            error_result = CalculationResult(
                expression=expression,
                result=None,
                result_type="error",
                method="python_eval",
                execution_time=time.time() - start_time,
                steps=steps,
                error=str(e)
            )
            
            self.history.append(error_result)
            self.logger.warning(f"Error evaluating expression '{expression}': {str(e)}")
            return error_result
    
    def _sanitize_expression(self, expression: str) -> str:
        """
        Sanitize an expression to remove potentially unsafe code.
        
        Args:
            expression: The expression to sanitize
            
        Returns:
            Sanitized expression
        """
        # Remove any import statements
        if "import" in expression:
            raise ValueError("Import statements are not allowed")
        
        # Check for potentially dangerous functions
        forbidden = ["exec", "eval", "compile", "globals", "locals", "getattr", "setattr", 
                    "__import__", "open", "file", "os", "sys", "subprocess"]
        for word in forbidden:
            if word in expression:
                raise ValueError(f"Forbidden function or module: {word}")
        
        return expression
    
    def _determine_result_type(self, result: Any) -> str:
        """
        Determine the type of a calculation result.
        
        Args:
            result: The result to analyze
            
        Returns:
            String describing the result type
        """
        if isinstance(result, (int, float)):
            return "numeric"
        elif isinstance(result, complex):
            return "complex"
        elif isinstance(result, bool):
            return "boolean"
        elif isinstance(result, str):
            return "string"
        elif isinstance(result, list):
            return "array"
        elif isinstance(result, tuple):
            return "tuple"
        elif isinstance(result, dict):
            return "dictionary"
        elif isinstance(result, np.ndarray):
            if result.ndim <= 1:
                return "array"
            else:
                return "matrix"
        else:
            return type(result).__name__
    
    def calculate_statistics(
        self,
        data: List[Union[int, float]],
        operations: List[str] = ["mean", "median", "stdev"]
    ) -> CalculationResult:
        """
        Calculate statistical measures for a dataset.
        
        Args:
            data: List of numeric values
            operations: Statistical operations to perform
            
        Returns:
            CalculationResult with statistical results
        """
        start_time = time.time()
        steps = []
        
        try:
            # Convert data to numpy array
            np_data = np.array(data, dtype=float)
            steps.append(f"Converted input data to numpy array of shape {np_data.shape}")
            
            # Initialize results dictionary
            stats_results = {}
            
            # Perform requested operations
            for op in operations:
                if op == "mean":
                    stats_results["mean"] = float(np.mean(np_data))
                    steps.append(f"Calculated mean: {stats_results['mean']}")
                
                elif op == "median":
                    stats_results["median"] = float(np.median(np_data))
                    steps.append(f"Calculated median: {stats_results['median']}")
                
                elif op == "mode":
                    try:
                        mode_result = statistics.mode(np_data)
                        stats_results["mode"] = float(mode_result)
                        steps.append(f"Calculated mode: {stats_results['mode']}")
                    except statistics.StatisticsError as e:
                        stats_results["mode"] = str(e)
                        steps.append(f"Mode calculation error: {e}")
                
                elif op == "stdev":
                    if len(np_data) > 1:
                        stats_results["stdev"] = float(np.std(np_data, ddof=1))
                        steps.append(f"Calculated standard deviation: {stats_results['stdev']}")
                    else:
                        stats_results["stdev"] = "N/A (need at least 2 data points)"
                        steps.append("Standard deviation requires at least 2 data points")
                
                elif op == "variance":
                    if len(np_data) > 1:
                        stats_results["variance"] = float(np.var(np_data, ddof=1))
                        steps.append(f"Calculated variance: {stats_results['variance']}")
                    else:
                        stats_results["variance"] = "N/A (need at least 2 data points)"
                        steps.append("Variance requires at least 2 data points")
                
                elif op == "min":
                    stats_results["min"] = float(np.min(np_data))
                    steps.append(f"Found minimum value: {stats_results['min']}")
                
                elif op == "max":
                    stats_results["max"] = float(np.max(np_data))
                    steps.append(f"Found maximum value: {stats_results['max']}")
                
                elif op == "range":
                    stats_results["range"] = float(np.max(np_data) - np.min(np_data))
                    steps.append(f"Calculated range: {stats_results['range']}")
                
                elif op == "sum":
                    stats_results["sum"] = float(np.sum(np_data))
                    steps.append(f"Calculated sum: {stats_results['sum']}")
                
                elif op == "count":
                    stats_results["count"] = int(len(np_data))
                    steps.append(f"Counted elements: {stats_results['count']}")
                
                elif op == "quartiles":
                    q1 = float(np.percentile(np_data, 25))
                    q2 = float(np.percentile(np_data, 50))
                    q3 = float(np.percentile(np_data, 75))
                    stats_results["quartiles"] = [q1, q2, q3]
                    steps.append(f"Calculated quartiles: Q1={q1}, Q2={q2}, Q3={q3}")
                
                elif op == "describe":
                    # Similar to pandas describe() function
                    count = len(np_data)
                    mean = float(np.mean(np_data))
                    std = float(np.std(np_data, ddof=1)) if count > 1 else "N/A"
                    min_val = float(np.min(np_data))
                    q1 = float(np.percentile(np_data, 25))
                    q2 = float(np.percentile(np_data, 50))
                    q3 = float(np.percentile(np_data, 75))
                    max_val = float(np.max(np_data))
                    
                    stats_results["describe"] = {
                        "count": count,
                        "mean": mean,
                        "std": std,
                        "min": min_val,
                        "25%": q1,
                        "50%": q2,
                        "75%": q3,
                        "max": max_val
                    }
                    steps.append("Calculated descriptive statistics summary")
            
            # Create and return the result
            calc_result = CalculationResult(
                expression=f"Statistical analysis of {len(np_data)} data points",
                result=stats_results,
                result_type="statistics",
                method="numpy_statistics",
                execution_time=time.time() - start_time,
                steps=steps
            )
            
            self.history.append(calc_result)
            self.logger.info(f"Calculated statistics for {len(data)} data points")
            return calc_result
        
        except Exception as e:
            # Handle any errors
            error_result = CalculationResult(
                expression=f"Statistical analysis of data",
                result=None,
                result_type="error",
                method="numpy_statistics",
                execution_time=time.time() - start_time,
                steps=steps,
                error=str(e)
            )
            
            self.history.append(error_result)
            self.logger.warning(f"Error calculating statistics: {str(e)}")
            return error_result
    
    def matrix_operation(
        self,
        operation: str,
        matrix_a: Union[List[List[float]], np.ndarray],
        matrix_b: Optional[Union[List[List[float]], np.ndarray]] = None,
        **kwargs
    ) -> CalculationResult:
        """
        Perform matrix operations.
        
        Args:
            operation: Type of operation (add, subtract, multiply, inverse, determinant, etc.)
            matrix_a: First matrix
            matrix_b: Second matrix (if needed for the operation)
            **kwargs: Additional operation-specific parameters
            
        Returns:
            CalculationResult with the result
        """
        start_time = time.time()
        steps = []
        
        try:
            # Convert matrices to numpy arrays if they aren't already
            if not isinstance(matrix_a, np.ndarray):
                matrix_a = np.array(matrix_a, dtype=float)
                steps.append(f"Converted matrix A to numpy array of shape {matrix_a.shape}")
            
            if matrix_b is not None and not isinstance(matrix_b, np.ndarray):
                matrix_b = np.array(matrix_b, dtype=float)
                steps.append(f"Converted matrix B to numpy array of shape {matrix_b.shape}")
            
            # Perform the requested operation
            result = None
            
            if operation == "add":
                if matrix_b is None:
                    raise ValueError("Matrix addition requires two matrices")
                result = matrix_a + matrix_b
                steps.append("Performed matrix addition")
            
            elif operation == "subtract":
                if matrix_b is None:
                    raise ValueError("Matrix subtraction requires two matrices")
                result = matrix_a - matrix_b
                steps.append("Performed matrix subtraction")
            
            elif operation == "multiply":
                if matrix_b is None:
                    raise ValueError("Matrix multiplication requires two matrices")
                result = np.matmul(matrix_a, matrix_b)
                steps.append(f"Performed matrix multiplication resulting in shape {result.shape}")
            
            elif operation == "inverse":
                result = np.linalg.inv(matrix_a)
                steps.append("Calculated matrix inverse")
            
            elif operation == "determinant":
                result = np.linalg.det(matrix_a)
                steps.append(f"Calculated determinant: {result}")
            
            elif operation == "transpose":
                result = np.transpose(matrix_a)
                steps.append(f"Calculated transpose resulting in shape {result.shape}")
            
            elif operation == "eigenvalues":
                result = np.linalg.eigvals(matrix_a)
                steps.append(f"Calculated eigenvalues: {result}")
            
            elif operation == "eigenvectors":
                eigenvalues, eigenvectors = np.linalg.eig(matrix_a)
                result = {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}
                steps.append("Calculated eigenvalues and eigenvectors")
            
            elif operation == "solve":
                if matrix_b is None:
                    raise ValueError("Matrix equation solving requires a coefficient matrix and a constants vector")
                result = np.linalg.solve(matrix_a, matrix_b)
                steps.append(f"Solved linear system, solution: {result}")
            
            elif operation == "rank":
                result = np.linalg.matrix_rank(matrix_a)
                steps.append(f"Calculated matrix rank: {result}")
            
            elif operation == "norm":
                ord_value = kwargs.get("ord", None)
                result = np.linalg.norm(matrix_a, ord=ord_value)
                steps.append(f"Calculated matrix norm: {result}")
            
            elif operation == "trace":
                result = np.trace(matrix_a)
                steps.append(f"Calculated matrix trace: {result}")
            
            elif operation == "svd":
                u, s, vh = np.linalg.svd(matrix_a)
                result = {"U": u, "singular_values": s, "Vh": vh}
                steps.append("Performed singular value decomposition")
            
            else:
                raise ValueError(f"Unknown matrix operation: {operation}")
            
            # Determine result type
            result_type = self._determine_result_type(result)
            
            # Create and return the result
            calc_result = CalculationResult(
                expression=f"Matrix operation: {operation}",
                result=result,
                result_type=result_type,
                method="numpy_linalg",
                execution_time=time.time() - start_time,
                steps=steps
            )
            
            self.history.append(calc_result)
            self.logger.info(f"Performed matrix operation: {operation}")
            return calc_result
        
        except Exception as e:
            # Handle any errors
            error_result = CalculationResult(
                expression=f"Matrix operation: {operation}",
                result=None,
                result_type="error",
                method="numpy_linalg",
                execution_time=time.time() - start_time,
                steps=steps,
                error=str(e)
            )
            
            self.history.append(error_result)
            self.logger.warning(f"Error in matrix operation '{operation}': {str(e)}")
            return error_result
    
    def numerical_method(
        self,
        method: str,
        function: Callable,
        **kwargs
    ) -> CalculationResult:
        """
        Apply numerical methods to solve problems.
        
        Args:
            method: Numerical method to use (integration, root_finding, etc.)
            function: Function to apply the method to
            **kwargs: Method-specific parameters
            
        Returns:
            CalculationResult with the numerical solution
        """
        start_time = time.time()
        steps = []
        
        try:
            result = None
            
            if method == "integration":
                # Numerical integration (quadrature)
                a = kwargs.get("lower_bound", 0)
                b = kwargs.get("upper_bound", 1)
                
                steps.append(f"Setting up numerical integration from {a} to {b}")
                
                # Use Simpson's rule by default
                n = kwargs.get("points", 1000)
                if n % 2 == 0:
                    n += 1  # Ensure odd number of points for Simpson's rule
                
                h = (b - a) / (n - 1)
                x = np.linspace(a, b, n)
                y = np.array([function(xi) for xi in x])
                
                # Apply Simpson's 1/3 rule
                weights = np.ones(n)
                weights[1:-1:2] = 4  # Odd indices (except endpoints)
                weights[2:-1:2] = 2  # Even indices (except endpoints)
                
                result = h / 3 * np.sum(weights * y)
                steps.append(f"Applied Simpson's rule with {n} points")
                steps.append(f"Calculated definite integral: {result}")
            
            elif method == "root_finding":
                # Numerical root finding
                x0 = kwargs.get("initial_guess", 0)
                tol = kwargs.get("tolerance", 1e-6)
                max_iter = kwargs.get("max_iterations", 100)
                
                steps.append(f"Starting root finding with initial guess x0 = {x0}")
                
                # Newton-Raphson method
                x = x0
                for i in range(max_iter):
                    # Approximate derivative
                    h = 1e-8
                    f_x = function(x)
                    f_x_h = function(x + h)
                    f_prime = (f_x_h - f_x) / h
                    
                    # Check if derivative is too small
                    if abs(f_prime) < 1e-14:
                        steps.append(f"Derivative too small at iteration {i}, stopping")
                        break
                    
                    # Update x
                    x_new = x - f_x / f_prime
                    steps.append(f"Iteration {i+1}: x = {x_new}, f(x) = {function(x_new)}")
                    
                    # Check convergence
                    if abs(x_new - x) < tol:
                        steps.append(f"Converged after {i+1} iterations")
                        break
                    
                    x = x_new
                
                result = x
                steps.append(f"Found root at x = {result} with f(x) = {function(result)}")
            
            elif method == "optimization":
                # Find minimum of a function
                x0 = kwargs.get("initial_guess", 0)
                tol = kwargs.get("tolerance", 1e-6)
                max_iter = kwargs.get("max_iterations", 100)
                
                steps.append(f"Starting optimization with initial guess x0 = {x0}")
                
                # Golden section search for 1D optimization
                a = kwargs.get("lower_bound", x0 - 10)
                b = kwargs.get("upper_bound", x0 + 10)
                
                golden_ratio = (1 + math.sqrt(5)) / 2
                
                # Initialize points for golden section search
                c = b - (b - a) / golden_ratio
                d = a + (b - a) / golden_ratio
                
                for i in range(max_iter):
                    if function(c) < function(d):
                        b = d
                    else:
                        a = c
                    
                    # Update points
                    c = b - (b - a) / golden_ratio
                    d = a + (b - a) / golden_ratio
                    
                    steps.append(f"Iteration {i+1}: interval [{a}, {b}]")
                    
                    # Check convergence
                    if abs(b - a) < tol:
                        steps.append(f"Converged after {i+1} iterations")
                        break
                
                # The minimum is approximately at the midpoint of the final interval
                result = (a + b) / 2
                steps.append(f"Found minimum at x = {result} with f(x) = {function(result)}")
            
            elif method == "differentiation":
                # Numerical differentiation
                x = kwargs.get("point", 0)
                h = kwargs.get("step_size", 1e-6)
                order = kwargs.get("order", 1)
                
                steps.append(f"Calculating derivative at x = {x} with step size h = {h}")
                
                if order == 1:
                    # Central difference for first derivative
                    result = (function(x + h) - function(x - h)) / (2 * h)
                    steps.append("Used central difference formula for first derivative")
                
                elif order == 2:
                    # Central difference for second derivative
                    result = (function(x + h) - 2 * function(x) + function(x - h)) / (h**2)
                    steps.append("Used central difference formula for second derivative")
                
                else:
                    raise ValueError(f"Differentiation of order {order} not supported")
                
                steps.append(f"Calculated derivative: {result}")
            
            elif method == "ode_solver":
                # Simple Euler method for ODEs
                t_span = kwargs.get("t_span", [0, 1])
                y0 = kwargs.get("initial_value", 0)
                num_steps = kwargs.get("num_steps", 100)
                
                t_start, t_end = t_span
                dt = (t_end - t_start) / num_steps
                
                steps.append(f"Solving ODE from t = {t_start} to t = {t_end} with {num_steps} steps")
                
                # Initialize arrays
                t = np.linspace(t_start, t_end, num_steps + 1)
                y = np.zeros(num_steps + 1)
                y[0] = y0
                
                # Euler method
                for i in range(num_steps):
                    y[i+1] = y[i] + dt * function(t[i], y[i])
                
                result = {"t": t, "y": y}
                steps.append(f"Applied Euler method, final value: {y[-1]}")
            
            else:
                raise ValueError(f"Unknown numerical method: {method}")
            
            # Determine result type
            result_type = self._determine_result_type(result)
            
            # Create and return the result
            calc_result = CalculationResult(
                expression=f"Numerical {method}",
                result=result,
                result_type=result_type,
                method=f"numerical_{method}",
                execution_time=time.time() - start_time,
                steps=steps
            )
            
            self.history.append(calc_result)
            self.logger.info(f"Applied numerical method: {method}")
            return calc_result
        
        except Exception as e:
            # Handle any errors
            error_result = CalculationResult(
                expression=f"Numerical {method}",
                result=None,
                result_type="error",
                method=f"numerical_{method}",
                execution_time=time.time() - start_time,
                steps=steps,
                error=str(e)
            )
            
            self.history.append(error_result)
            self.logger.warning(f"Error in numerical method '{method}': {str(e)}")
            return error_result
    
    def symbolic_math(self, expression: str, operation: str, **kwargs) -> CalculationResult:
        """
        Perform symbolic mathematics operations.
        
        Args:
            expression: Mathematical expression to operate on
            operation: Symbolic operation to perform (differentiate, integrate, solve, etc.)
            **kwargs: Additional operation-specific parameters
            
        Returns:
            CalculationResult with the symbolic result
        """
        start_time = time.time()
        steps = []
        
        if not SYMPY_AVAILABLE:
            return CalculationResult(
                expression=expression,
                result=None,
                result_type="error",
                method="symbolic_math",
                execution_time=time.time() - start_time,
                steps=steps,
                error="SymPy is not available. Install it to use symbolic mathematics."
            )
        
        try:
            # Parse the expression with SymPy
            steps.append(f"Parsing expression: {expression}")
            
            # Define common symbols
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            z = sympy.Symbol('z')
            t = sympy.Symbol('t')
            
            # Define a namespace for parsing
            namespace = {
                'x': x, 'y': y, 'z': z, 't': t,
                'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
                'exp': sympy.exp, 'log': sympy.log, 'sqrt': sympy.sqrt,
                'pi': sympy.pi, 'E': sympy.E, 'I': sympy.I
            }
            
            # Add any additional symbols from kwargs
            if "symbols" in kwargs:
                for sym_name in kwargs["symbols"]:
                    if sym_name not in namespace:
                        namespace[sym_name] = sympy.Symbol(sym_name)
            
            # Parse the expression
            sympy_expr = sympy.sympify(expression, locals=namespace)
            steps.append(f"Parsed as: {sympy_expr}")
            
            # Perform the requested operation
            result = None
            
            if operation == "simplify":
                result = sympy.simplify(sympy_expr)
                steps.append(f"Simplified expression: {result}")
            
            elif operation == "expand":
                result = sympy.expand(sympy_expr)
                steps.append(f"Expanded expression: {result}")
            
            elif operation == "factor":
                result = sympy.factor(sympy_expr)
                steps.append(f"Factored expression: {result}")
            
            elif operation == "differentiate":
                var_name = kwargs.get("variable", "x")
                var = namespace.get(var_name, sympy.Symbol(var_name))
                order = kwargs.get("order", 1)
                
                result = sympy.diff(sympy_expr, var, order)
                steps.append(f"Differentiated expression with respect to {var_name}: {result}")
            
            elif operation == "integrate":
                var_name = kwargs.get("variable", "x")
                var = namespace.get(var_name, sympy.Symbol(var_name))
                
                if "limits" in kwargs:
                    a, b = kwargs["limits"]
                    result = sympy.integrate(sympy_expr, (var, a, b))
                    steps.append(f"Computed definite integral from {a} to {b}: {result}")
                else:
                    result = sympy.integrate(sympy_expr, var)
                    steps.append(f"Computed indefinite integral: {result}")
            
            elif operation == "solve":
                var_name = kwargs.get("variable", "x")
                var = namespace.get(var_name, sympy.Symbol(var_name))
                
                result = sympy.solve(sympy_expr, var)
                steps.append(f"Solved equation for {var_name}: {result}")
            
            elif operation == "limit":
                var_name = kwargs.get("variable", "x")
                var = namespace.get(var_name, sympy.Symbol(var_name))
                point = kwargs.get("point", 0)
                direction = kwargs.get("direction", "+")
                
                result = sympy.limit(sympy_expr, var, point, direction)
                steps.append(f"Computed limit as {var_name} approaches {point}: {result}")
            
            elif operation == "series":
                var_name = kwargs.get("variable", "x")
                var = namespace.get(var_name, sympy.Symbol(var_name))
                point = kwargs.get("point", 0)
                order = kwargs.get("order", 6)
                
                result = sympy.series(sympy_expr, var, point, order).removeO()
                steps.append(f"Computed series expansion around {point} to order {order}: {result}")
            
            elif operation == "latex":
                result = sympy.latex(sympy_expr)
                steps.append(f"Generated LaTeX representation")
            
            else:
                raise ValueError(f"Unknown symbolic operation: {operation}")
            
            # Determine result type
            result_type = "symbolic"
            
            # Create and return the result
            calc_result = CalculationResult(
                expression=expression,
                result=str(result),  # Convert to string for serialization
                result_type=result_type,
                method=f"symbolic_{operation}",
                execution_time=time.time() - start_time,
                steps=steps
            )
            
            self.history.append(calc_result)
            self.logger.info(f"Performed symbolic operation: {operation}")
            return calc_result
        
        except Exception as e:
            # Handle any errors
            error_result = CalculationResult(
                expression=expression,
                result=None,
                result_type="error",
                method=f"symbolic_{operation}",
                execution_time=time.time() - start_time,
                steps=steps,
                error=str(e)
            )
            
            self.history.append(error_result)
            self.logger.warning(f"Error in symbolic operation '{operation}': {str(e)}")
            return error_result
    
    def convert_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> CalculationResult:
        """
        Convert between different units of measurement.
        
        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            CalculationResult with the converted value
        """
        start_time = time.time()
        steps = []
        
        try:
            # Define conversion factors to SI units
            length_to_meters = {
                "m": 1.0,
                "km": 1000.0,
                "cm": 0.01,
                "mm": 0.001,
                "in": 0.0254,
                "ft": 0.3048,
                "yd": 0.9144,
                "mi": 1609.344
            }
            
            mass_to_kg = {
                "kg": 1.0,
                "g": 0.001,
                "mg": 1e-6,
                "lb": 0.45359237,
                "oz": 0.028349523125,
                "tonne": 1000.0,
                "ton": 907.18474
            }
            
            time_to_seconds = {
                "s": 1.0,
                "min": 60.0,
                "h": 3600.0,
                "day": 86400.0,
                "week": 604800.0,
                "ms": 0.001,
                "us": 1e-6,
                "ns": 1e-9
            }
            
            temperature_conversions = {
                "C_to_F": lambda c: c * 9/5 + 32,
                "F_to_C": lambda f: (f - 32) * 5/9,
                "C_to_K": lambda c: c + 273.15,
                "K_to_C": lambda k: k - 273.15,
                "F_to_K": lambda f: (f - 32) * 5/9 + 273.15,
                "K_to_F": lambda k: (k - 273.15) * 9/5 + 32
            }
            
            area_to_sq_meters = {
                "m2": 1.0,
                "cm2": 0.0001,
                "km2": 1e6,
                "ft2": 0.09290304,
                "in2": 0.00064516,
                "acre": 4046.8564224,
                "hectare": 10000.0
            }
            
            volume_to_cubic_meters = {
                "m3": 1.0,
                "cm3": 1e-6,
                "mm3": 1e-9,
                "liter": 0.001,
                "ml": 1e-6,
                "gallon": 0.00378541,
                "quart": 0.000946353,
                "pint": 0.000473176,
                "cup": 0.000236588,
                "fl_oz": 2.95735e-5,
                "ft3": 0.0283168,
                "in3": 1.63871e-5
            }
            
            # Determine conversion category
            if from_unit in length_to_meters and to_unit in length_to_meters:
                # Length conversion
                meters = value * length_to_meters[from_unit]
                result = meters / length_to_meters[to_unit]
                steps.append(f"Converted {value} {from_unit} to {meters} meters")
                steps.append(f"Converted {meters} meters to {result} {to_unit}")
            
            elif from_unit in mass_to_kg and to_unit in mass_to_kg:
                # Mass conversion
                kg = value * mass_to_kg[from_unit]
                result = kg / mass_to_kg[to_unit]
                steps.append(f"Converted {value} {from_unit} to {kg} kg")
                steps.append(f"Converted {kg} kg to {result} {to_unit}")
            
            elif from_unit in time_to_seconds and to_unit in time_to_seconds:
                # Time conversion
                seconds = value * time_to_seconds[from_unit]
                result = seconds / time_to_seconds[to_unit]
                steps.append(f"Converted {value} {from_unit} to {seconds} seconds")
                steps.append(f"Converted {seconds} seconds to {result} {to_unit}")
            
            elif from_unit in area_to_sq_meters and to_unit in area_to_sq_meters:
                # Area conversion
                sq_meters = value * area_to_sq_meters[from_unit]
                result = sq_meters / area_to_sq_meters[to_unit]
                steps.append(f"Converted {value} {from_unit} to {sq_meters} square meters")
                steps.append(f"Converted {sq_meters} square meters to {result} {to_unit}")
            
            elif from_unit in volume_to_cubic_meters and to_unit in volume_to_cubic_meters:
                # Volume conversion
                cubic_meters = value * volume_to_cubic_meters[from_unit]
                result = cubic_meters / volume_to_cubic_meters[to_unit]
                steps.append(f"Converted {value} {from_unit} to {cubic_meters} cubic meters")
                steps.append(f"Converted {cubic_meters} cubic meters to {result} {to_unit}")
            
            elif (from_unit == "C" and to_unit == "F") or (from_unit == "F" and to_unit == "C") or \
                 (from_unit == "C" and to_unit == "K") or (from_unit == "K" and to_unit == "C") or \
                 (from_unit == "F" and to_unit == "K") or (from_unit == "K" and to_unit == "F"):
                # Temperature conversion
                conversion_key = f"{from_unit}_to_{to_unit}"
                result = temperature_conversions[conversion_key](value)
                steps.append(f"Applied temperature conversion formula from {from_unit} to {to_unit}")
            
            else:
                raise ValueError(f"Conversion from {from_unit} to {to_unit} is not supported")
            
            # Create and return the result
            calc_result = CalculationResult(
                expression=f"Convert {value} {from_unit} to {to_unit}",
                result=result,
                result_type="numeric",
                method="unit_conversion",
                execution_time=time.time() - start_time,
                steps=steps
            )
            
            self.history.append(calc_result)
            self.logger.info(f"Converted {value} {from_unit} to {to_unit}")
            return calc_result
        
        except Exception as e:
            # Handle any errors
            error_result = CalculationResult(
                expression=f"Convert {value} {from_unit} to {to_unit}",
                result=None,
                result_type="error",
                method="unit_conversion",
                execution_time=time.time() - start_time,
                steps=steps,
                error=str(e)
            )
            
            self.history.append(error_result)
            self.logger.warning(f"Error in unit conversion: {str(e)}")
            return error_result
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the recent calculation history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of calculation results as dictionaries
        """
        return [result.to_dict() for result in self.history[-limit:]]
    
    def save_history(self, file_path: str) -> bool:
        """
        Save calculation history to a file.
        
        Args:
            file_path: Path to the output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert history to serializable format
            history_data = [result.to_dict() for result in self.history]
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            self.logger.info(f"Saved calculation history to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving calculation history: {str(e)}")
            return False
    
    def load_history(self, file_path: str) -> bool:
        """
        Load calculation history from a file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"History file not found: {file_path}")
                return False
            
            # Load from file
            with open(file_path, 'r') as f:
                history_data = json.load(f)
            
            # Convert to CalculationResult objects
            self.history = []
            for item in history_data:
                result = CalculationResult(
                    expression=item["expression"],
                    result=item["result"],
                    result_type=item["result_type"],
                    method=item["method"],
                    execution_time=item["execution_time"],
                    steps=item.get("steps", []),
                    error=item.get("error")
                )
                result.id = item["id"]
                result.timestamp = item["timestamp"]
                self.history.append(result)
            
            self.logger.info(f"Loaded {len(self.history)} calculation history items from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading calculation history: {str(e)}")
            return False
    
    def clear_history(self) -> None:
        """Clear the calculation history."""
        self.history = []
        self.logger.info("Cleared calculation history")
    
    def format_result_as_string(self, result: CalculationResult) -> str:
        """
        Format a calculation result as a human-readable string.
        
        Args:
            result: The calculation result to format
            
        Returns:
            Formatted string
        """
        if not result.success:
            return f"Error: {result.error}"
        
        output = f"Expression: {result.expression}\n"
        output += f"Result: {result.result}\n"
        output += f"Type: {result.result_type}\n"
        output += f"Method: {result.method}\n"
        output += f"Execution time: {result.execution_time:.6f} seconds\n"
        
        if result.steps:
            output += "\nSteps:\n"
            for i, step in enumerate(result.steps, 1):
                output += f"{i}. {step}\n"
        
        return output


# Factory function for creating calculator tools
def create_calculator_tool(
    enable_symbolic: bool = False,
    logger: Optional[AgentLogger] = None
) -> CalculatorTool:
    """
    Create a CalculatorTool instance with the specified configuration.
    
    Args:
        enable_symbolic: Whether to enable symbolic mathematics (requires SymPy)
        logger: Logger instance
        
    Returns:
        Configured CalculatorTool instance
    """
    # Install SymPy if requested but not available
    if enable_symbolic and not SYMPY_AVAILABLE:
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "sympy"])
            import sympy
            global SYMPY_AVAILABLE
            SYMPY_AVAILABLE = True
        except Exception as e:
            if logger:
                logger.warning(f"Failed to install SymPy: {str(e)}")
            else:
                print(f"Warning: Failed to install SymPy: {str(e)}")
    
    # Create and return the calculator tool
    return CalculatorTool(logger=logger)
        