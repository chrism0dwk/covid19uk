import numpy as np


def simulate(initial_state, beta, nu, gamma):
    """Runs a discrete-time SEIR model with homogeneous mixing
    :param initial_state: a vector of length 4 giving initial SEIR values
    :param beta: infection rate
    :param nu: E->I rate
    :param gamma: I->R rate
    :returns: a tuple of events and states
    """
    params = np.array([beta, nu, gamma])

    def propagate(state):
        rates = params * np.array([state[2]/np.sum(state), 1., 1.])
        probs = 1. - np.exp(-rates)  # time step of 1
        events = np.random.binomial(state[:-1].astype(np.int64), probs)
        state[:-1] -= events
        state[1:] += events
        return state, events

    state_accum = []
    events_accum = []

    state = initial_state.copy()
    state_accum.append(state.copy())
    while(state[1:3].sum() > 0):
        state, events = propagate(state)
        state_accum.append(state.copy())
        events_accum.append(events.copy())

    return np.stack(state_accum), np.stack(events_accum)


def state_from_events(initial_state, events):
    state_accum = [initial_state]
    state = initial_state.copy()
    for event in events:
        state[:-1] -= event
        state[1:] += event
        state_accum.append(state.copy())
    return np.stack(state_accum)
