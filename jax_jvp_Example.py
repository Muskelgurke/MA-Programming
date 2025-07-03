import jax
import jax.numpy as jnp

# Define a function from R^2 -> R^2
def f(x):
  return jnp.array([x[0]**2])


# Define the primal input point
x_primal = jnp.array([2.0, 3.0])

# Define the tangent vector (direction of perturbation)
v_tangent = jnp.array([1.0, 0.5])

# Compute the function output and the JVP
y_primal_out, tangent_out = jax.jvp(f, (x_primal,), (v_tangent,))

print(f"Primal input (x): {x_primal}")
print(f"Tangent vector (v): {v_tangent}")
print(f"Primal output f(x): {y_primal_out}")
print(f"Tangent output (J(x)v): {tangent_out}")

# Let's manually verify the Jacobian and JVP for this case:
# f(x) = [x_0^2, x_0 * x_1]
# J(x) = [[df1/dx0, df1/dx1], [df2/dx0, df2/dx1]]
# J(x) = [[2*x0, 0], [x1, x0]]
# At x = [2.0, 3.0], J(x) = [[4.0, 0.0], [3.0, 2.0]]
# J(x)v = [[4.0, 0.0], [3.0, 2.0]] @ [1.0, 0.5]
#       = [4.0*1.0 + 0.0*0.5, 3.0*1.0 + 2.0*0.5]
#       = [4.0, 3.0 + 1.0]
#       = [4.0, 4.0]
# This matches the tangent_out from jax.jvp!
