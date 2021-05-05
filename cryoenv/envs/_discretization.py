import numpy as np


def action_to_discrete(reset: np.ndarray, V_decrease: np.ndarray, wait: np.ndarray,
                       wait_iv: tuple = (2, 100), V_iv: tuple = (0, 99), wait_step=2, V_step=1):
    assert np.round(wait / wait_step * 10e6) / 10e6 % 1 == 0, "wait has to be multiple of {}".format(
        wait_step)
    assert wait >= wait_iv[0] and wait <= wait_iv[1], "wait should be between {} and {}".format(*wait_iv)
    assert all(type(r) == np.bool_ for r in reset), "reset has to be bool variable"
    assert all(
        np.round(v / V_step * 10e6) / 10e6 % 1 == 0 for v in V_decrease), "V_decrease has to be multiple of {}".format(
        V_step)
    assert all(v >= V_iv[0] and v <= V_iv[1] for v in V_decrease), "V_decrease should be between {} and {}".format(
        *V_iv)

    nmbr_channels = len(reset)
    m = int((V_iv[1] - V_iv[0]) / V_step + 2)  # nmbr discrete dV and the reset per channel
    k = int((wait_iv[1] - wait_iv[0]) / wait_step + 1)  # nmbr discrete wait (for all channels)
    nmbr_n = k * m ** nmbr_channels

    # print(nmbr_channels, m, k, nmbr_n)

    m_ = []

    for i in range(nmbr_channels):
        if reset[i] == True:
            m_.append(m - 1)
        else:
            m_.append((V_decrease[i] - V_iv[0]) / V_step)

    # print('m_: ', m_)

    k_ = (wait - wait_iv[0]) / wait_step
    temp = np.sum([m_[i] * m ** i for i in range(nmbr_channels)])
    # print('temp: ', temp)
    n = k_ * m ** nmbr_channels + temp
    # print('n: ', n)

    assert n < nmbr_n, 'The calculated n is larger than the number of n - check the conversion function!'
    return n


def action_from_discrete(n, nmbr_channels, wait_iv=(2, 100), V_iv=(0, 99), wait_step=2, V_step=1):
    m = int((V_iv[1] - V_iv[0]) / V_step + 2)  # nmbr discrete dV and the reset per channel
    k = int((wait_iv[1] - wait_iv[0]) / wait_step + 1)  # nmbr discrete wait (for all channels)
    nmbr_n = k * m ** nmbr_channels

    assert n % 1 == 0, "n has to be multiple of 1"
    assert n >= 0 and n < nmbr_n, "n should be between 0 and {}".format(nmbr_n)

    wait = int(n / m ** nmbr_channels)*wait_step + wait_iv[0]
    # print('wait: ', wait)
    temp = n % m ** nmbr_channels
    # print('temp: ', temp)
    m_ = [int(temp % m ** (i + 1) / m ** i) for i in range(nmbr_channels)]
    # print('m_: ', m_)

    reset = []
    V_decrease = []

    for val in m_:
        if val == m-1:
            # first action is reset, second V_decrease, third wait
            reset.append(True)
            V_decrease.append(0)
        else:
            reset.append(False)
            V_decrease.append(val * V_step + V_iv[0])

    return np.array(reset), np.array(V_decrease), np.array(wait)


def observation_to_discrete(V_set: np.ndarray, ph: np.ndarray, V_iv=(0, 99), ph_iv=(0, 0.99), V_step=1, ph_step=0.01):
    assert all(np.round(p / ph_step * 10e6) / 10e6 % 1 == 0 for p in ph), "ph has to be multiple of {}".format(ph_step)
    assert all(p >= ph_iv[0] and p <= ph_iv[1] for p in ph), "ph should be between {} and {}".format(*ph_iv)
    assert all(np.round(v / V_step * 10e6) / 10e6 % 1 == 0 for v in V_set), "V_set has to be multiple of {}".format(
        V_step)
    assert all(v >= V_iv[0] and v <= V_iv[1] for v in V_set), "V_set should be between {} and {}".format(*V_iv)

    nmbr_discrete_V = int((V_iv[1] - V_iv[0]) / V_step + 1)
    nmbr_discrete_ph = int((ph_iv[1] - ph_iv[0]) / ph_step + 1)
    # n_max = int(nmbr_discrete_V * nmbr_discrete_ph - 1)
    len_n = int(nmbr_discrete_V * nmbr_discrete_ph)

    ns = []

    for v, p in zip(V_set, ph):
        V_in_range = (v - V_iv[0]) / V_step  # in (0, nmbr_discrete_V - 1)
        ph_in_range = (p - ph_iv[0]) / ph_step  # in (0, nmbr_discrete_ph - 1)

        ns.append(int(V_in_range * nmbr_discrete_ph + ph_in_range))

    # print('ns: ', ns)
    n = int(np.sum([n * len_n ** i for i, n in enumerate(ns)]))
    # print('n: ', n)
    return n


def observation_from_discrete(n, nmbr_channels, V_iv=(0, 99), ph_iv=(0, 0.99), V_step=1, ph_step=0.01):
    nmbr_discrete_V = int((V_iv[1] - V_iv[0]) / V_step + 1)
    nmbr_discrete_ph = int((ph_iv[1] - ph_iv[0]) / ph_step + 1)
    # n_max = int(nmbr_discrete_V * nmbr_discrete_ph - 1)  # this is length - 1
    len_n = int(nmbr_discrete_V * nmbr_discrete_ph)

    assert n % 1 == 0, "n has to be multiple of 1"
    assert n >= 0 and n <= len_n ** nmbr_channels, "n should be between 0 and {}".format(len_n ** nmbr_channels)

    ns = [int(n % len_n ** (i + 1) / len_n ** i) for i in range(nmbr_channels)]
    # print('ns: ', ns)

    V_set = []
    pulse_height = []

    for n in ns:
        V_set.append(np.floor(n / nmbr_discrete_ph) * V_step + V_iv[0])
        pulse_height.append((n % nmbr_discrete_V) * ph_step + ph_iv[0])

    return np.array(V_set), np.array(pulse_height)
