import numpy as np

def return_MTU_velocity(state_equations,MA_equations):
    """
    MTU velocity equation for a simple hinge joint with dynamics given by a simple pendulum equation,

               ẍ = F(x,ẋ)
                   ↓
               ⎡ x₁ = x ⎤
               ⎣ x₂ = ẋ ⎦
                   ↓
      ⎡   ẋ₁ = x₂ = f₁(x₁,x₂)     ⎤
      ⎣ ẋ₂ = F(x₁,x₂) = f₂(x₁,x₂) ⎦

    where f₁ and f₂ are the state_equations.

    MA_equations should be a list of length 3 and contain the moment arm function as well as first and second derivative w.r.t. joint angle (x₁). Each function should only take in one value (x₁), but the resulting functions will be of all state variables (X) - as long as X has 2 or more states.

    returns MTU velocity equation that is a functions of state array X (of length >= 2).

    NOTE: this value is not normalized (given in meters).
    """
    assert (len(state_equations) == 2
            and type(state_equations) == list), \
        "state_equations must be a list of length 2."
    assert all([str(type(el))=="<class 'function'>"
            for el in state_equations]), \
        "All elements in state_equations must be functions."

    assert (len(MA_equations) == 3
            and type(MA_equations) == list), \
        "MA_equations must be a list of length 2."
    assert all([str(type(el))=="<class 'function'>"
            for el in MA_equations]), \
        "All elements in MA_equations must be functions."

    ẋ1 = state_equations[0]
    ẋ2 = state_equations[1]
    r = MA_equations[0]
    dr_dx1 = MA_equations[1]
    d2r_dx12 = MA_equations[2]

    def v_MTU(X):
        return(
            np.sign(-r(X[0]))
            * ẋ1(X)
            * np.sqrt(dr_dx1(X[0])**2 + r(X[0])**2)
            )

    return(v_MTU)

def return_MTU_acceleration(state_equations,MA_equations):
    """
    MTU acceleration equation for a simple hinge joint with dynamics given by a simple pendulum equation,

               ẍ = F(x,ẋ)
                   ↓
               ⎡ x₁ = x ⎤
               ⎣ x₂ = ẋ ⎦
                   ↓
      ⎡   ẋ₁ = x₂ = f₁(x₁,x₂)     ⎤
      ⎣ ẋ₂ = F(x₁,x₂) = f₂(x₁,x₂) ⎦

    where f₁ and f₂ are the state_equations.

    MA_equations should be a list of length 3 and contain the moment arm function as well as first and second derivative w.r.t. joint angle (x₁). Each function should only take in one value (x₁), but the resulting functions will be of all state variables (X) - as long as X has 2 or more states.

    returns MTU acceleration equation that is a functions of state array X (of length >= 2).

    NOTE: this value is not normalized (given in meters).
    """
    assert (len(state_equations) == 2
            and type(state_equations) == list), \
        "state_equations must be a list of length 2."
    assert all([str(type(el))=="<class 'function'>"
            for el in state_equations]), \
        "All elements in state_equations must be functions."

    assert (len(MA_equations) == 3
            and type(MA_equations) == list), \
        "MA_equations must be a list of length 2."
    assert all([str(type(el))=="<class 'function'>"
            for el in MA_equations]), \
        "All elements in MA_equations must be functions."

    ẋ1 = state_equations[0]
    ẋ2 = state_equations[1]
    r = MA_equations[0]
    dr_dx1 = MA_equations[1]
    d2r_dx12 = MA_equations[2]

    def a_MTU(X):
    	return(
            np.sign(-r(X[0]))*(
                ẋ2(X)
                * np.sqrt(dr_dx1(X[0])**2 + r(X[0])**2)
    			+
                ẋ1(X)**2
                * dr_dx1(X[0])
                * (d2r_dx12(X[0]) + r(X[0]))
                / np.sqrt(dr_dx1(X[0])**2 + r(X[0])**2)
                )
            )

    return(a_MTU)
