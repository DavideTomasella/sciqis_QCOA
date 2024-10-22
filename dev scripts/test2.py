# In[]
from sympy import symbols, Function, Eq, dsolve,solve, Derivative, pprint

# Define the symbols
t = symbols('t')
x = Function('x')(t)
y = Function('y')(t)
z = Function('z')(t)
a, b, c, d, e = symbols('a b c d e')

# Define the equations
eq1 = Eq(Derivative(x, t), a*x - y*z)
eq2 = Eq(Derivative(y, t), b*y - x*z)
eq3 = Eq(Derivative(z, t), c*z - y*x)
# Solve the system of equations
#solution = dsolve([eq1, eq2, eq3])

#steady state solution solver
d_eq1 = Eq(0, a*x - y)
d_eq2 = Eq(0, b*y - x*z+d)
d_eq3 = Eq(0, c*z - y*x+e)
# Solve the system of equations
solution = solve([d_eq1, d_eq2, d_eq3], (x, y, z))
pprint(solution,wrap_line=False)
# Print the solution
print(solution)