from mici.adapters import Adapter
from mici.errors import IntegratorError, AdaptationError


class ToleranceAdapter(Adapter):
    """Adapter for projection solver tolerances.

    Allows setting coarser tolerances during adaptive warm-up phase of chains before
    using tighter tolerances for main chain iterations used to generate traces. This
    can help avoid unneccessary computational effort in early warm-up iterations.

    At the end of warm-up it is attempted to project the state to within the tighter
    tolerances prior to starting main iterations; if this projection fails an
    `AdaptationError` will be raised.
    """

    is_fast = True

    def __init__(
        self,
        warm_up_constraint_tol=1e-6,
        warm_up_position_tol=1e-5,
        warm_up_reverse_check_tol=None,
        main_constraint_tol=1e-9,
        main_position_tol=1e-8,
        main_reverse_check_tol=None,
    ):
        self.warm_up_constraint_tol = warm_up_constraint_tol
        self.warm_up_position_tol = warm_up_position_tol
        self.warm_up_reverse_check_tol = (
            2 * warm_up_position_tol
            if warm_up_reverse_check_tol is None
            else warm_up_reverse_check_tol
        )
        self.main_constraint_tol = main_constraint_tol
        self.main_position_tol = main_position_tol
        self.main_reverse_check_tol = (
            2 * main_position_tol
            if main_reverse_check_tol is None
            else main_reverse_check_tol
        )

    def initialize(self, chain_state, transition):
        transition.integrator.projection_solver_kwargs = {
            "constraint_tol": self.warm_up_constraint_tol,
            "position_tol": self.warm_up_position_tol,
        }
        transition.integrator.reverse_check_tol = self.warm_up_reverse_check_tol
        return {}

    def update(self, adapt_state, chain_state, trans_stats, transition):
        pass

    def finalize(self, adapt_states, chain_states, transition, rngs):
        integrator = transition.integrator
        system = transition.system
        integrator.projection_solver_kwargs = {
            "constraint_tol": self.main_constraint_tol,
            "position_tol": self.main_position_tol,
        }
        for state in chain_states:
            try:
                state = integrator.projection_solver(
                    state,
                    state,
                    integrator.step_size,
                    system,
                    **integrator.projection_solver_kwargs,
                )
            except IntegratorError as e:
                raise AdaptationError(
                    "Could not project within specified tolerances for main phase."
                ) from e
        transition.integrator.reverse_check_tol = self.main_reverse_check_tol
