import jax.numpy as jnp
import interpax

# these two methods wrap around interpax, and provide two ways of making the interpolator (extrapolator?) decay when extrapolated

def interp_decay(old_x, coords, values, method='cubic2', decay_rate=15.):
    value_left = values[coords.argmin()]
    value_right = values[coords.argmax()]
    values_interp = jnp.select([old_x < coords.min(), 
                                jnp.logical_and(coords.min() <= old_x, old_x <= coords.max()),
                                coords.max() < old_x],
                                [jnp.exp(-decay_rate * jnp.abs(old_x - coords.min())) * value_left,
                                interpax.interp1d(old_x, coords, values, method=method),
                                jnp.exp(-decay_rate * jnp.abs(old_x - coords.max())) * value_right])
    return values_interp

def interp_decay_knots(old_x, coords, values, method='catmull-rom', padding_percentage=0.1, extrap=1e-6):
    padding = padding_percentage * (coords.max() - coords.min())
    coords_extended = jnp.concatenate((jnp.array([coords.min() - padding]), coords, jnp.array([coords.max() + padding])))
    values_extended = jnp.concatenate((jnp.array([1e-6]), values, jnp.array([1e-6])))
    return interpax.interp1d(old_x, coords_extended, values_extended, method=method, extrap=extrap)

