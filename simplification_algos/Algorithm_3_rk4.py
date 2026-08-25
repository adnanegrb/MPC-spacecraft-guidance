def rk4_step(g, t, y, h, *args):
    k1 = g(t, y, *args)
    y2 = [y[i] + h / 2 * k1[i] for i in range(len(y))]
    k2 = g(t + h / 2, y2, *args)
    y3 = [y[i] + h / 2 * k2[i] for i in range(len(y))]
    k3 = g(t + h / 2, y3, *args)
    y4 = [y[i] + h * k3[i] for i in range(len(y))]
    k4 = g(t + h, y4, *args)
    return [y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(len(y))]


def rk4_integrate(g, t0, y0, h, n_steps, *args):
    t = t0
    y = y0[:]
    trajectory = [y[:]]
    for _ in range(n_steps):
        y = rk4_step(g, t, y, h, *args)
        t += h
        trajectory.append(y[:])
    return trajectory
