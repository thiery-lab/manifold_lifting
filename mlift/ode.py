import numpy as onp
import jax.numpy as np
import jax.lax as lax
from jax.experimental.ode import odeint


def integrate_ode_rk4(dx_dt, x_init, t_seq, params, dt):
    """Integrate ordinary differential equation using fourth-order Runge-Kutta method.

    Args:
        dx_dt (Callable[[ArrayLike, float, Dict], ArrayLike]): Function specifying the
            time-derivative of the state `x` at time `t`. First argument is current
            state `x` as an array, second argument current time `t` and third argument
            a dictionary of model parameters. Returns an array corresponding to
            time-derivative at `(x, t)`.
        x_init (ArrayLike): Initial state of system at time `t_seq[0]`.
        t_seq (ArrayLike): Sequence of time points to compute solution at. Must be
            static / known at compile time.
        params (Dict): Dictionary of model parameters to pass to `dx_dt` function.
        dt (float): Fixed time step to use in RK4 integrator. Must be static / known at
            compile time.

    Returns:
        ArrayLike: Sequence of states computed at times in `t_seq`.
    """

    def rk4_step_func(x_t, dt):
        x, t = x_t
        k1 = dx_dt(x, t, params)
        k2 = dx_dt(x + dt * k1 / 2, t + dt / 2, params)
        k3 = dx_dt(x + dt * k2 / 2, t + dt / 2, params)
        k4 = dx_dt(x + dt * k3, t + dt, params)
        x_ = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return (x_, t + dt), x_

    dt_seq, indices = _compute_dt_seq(t_seq, dt)

    _, x_seq = lax.scan(rk4_step_func, (x_init, t_seq[0]), dt_seq)

    return np.concatenate((x_init[None], x_seq[indices]))


def _compute_dt_seq(t_seq, dt):
    diff_t_seq = t_seq[1:] - t_seq[:-1]
    full_steps, remainders = onp.divmod(diff_t_seq, dt)
    full_steps = full_steps.astype(onp.int64)
    full_steps[remainders > 0] += 1
    cumulative_steps = full_steps.cumsum()
    dt_seq = onp.ones(cumulative_steps[-1]) * dt
    dt_seq[(cumulative_steps - 1)[remainders > 0]] = remainders[remainders > 0]
    return dt_seq, cumulative_steps - 1


def integrate_ode_adaptive(
    dx_dt, x_init, t_seq, params, rtol=1.4e-8, atol=1.4e-8, max_steps=np.inf
):
    """Integrate ordinary differential equation using adaptive Runge-Kutta method.

    Args:
        dx_dt (Callable[[ArrayLike, float, Dict], ArrayLike]): Function specifying the
            time-derivative of the state `x` at time `t`. First argument is current
            state `x` as an array, second argument current time `t` and third argument
            a dictionary of model parameters. Returns an array corresponding to
            time-derivative at `(x, t)`.
        x_init (ArrayLike): Initial state of system at time `t_seq[0]`.
        t_seq (ArrayLike): Sequence of time points to compute solution at.
        params (Dict): Dictionary of model parameters to pass to `dx_dt` function.
        rtol (float): Relative local error tolerance for solver.
        atol (float): Absolute local error tolerance for solver.
        max_steps (int): Kaximum number of steps to take between each time point.

    Returns:
        ArrayLike: Sequence of states computed at times in `t_seq`.
    """
    return odeint(dx_dt, x_init, t_seq, params, rtol=rtol, atol=atol, mxstep=max_steps)
