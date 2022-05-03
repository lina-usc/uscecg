
# Copied and modified from https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
#  # the cardiac equations are given by 2 ode's

# ode for x1-SV

from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import expit

def threshold_fct(x, c=10):
    return expit(c*x)


def get_fhn_currents(y1, y3_hp, k_at_de, k_at_re, k_vn_de, k_vn_re):
    # Ensuring differentiability by replacing the discontinuity
    # by a continuous function.

    # I_at_de = np.where(k_I[0]*y1, y1> 0, 0)
    I_at_de = k_at_de *y1 *threshold_fct(y1)

    # I_at_re = np.where(-k_I[1]*y1, y1<= 0, 0)
    I_at_re = -k_at_re *y1 *threshold_fct(-y1)

    # I_vn_de = np.where(k_I[2]*0.5*y3_hp, y3_hp > 0, 0)
    I_vn_de = k_vn_de *0.5 *y3_hp *threshold_fct(y3_hp)

    # I_vn_re = np.where(-k_I[3]*0.5*y3_hp, y3_hp <= 0, 0)
    I_vn_re = -k_vn_re *0.5 *y3_hp *threshold_fct(-y3_hp)

    return [I_at_de, I_at_re, I_vn_de, I_vn_re]


def get_fitzhugo_nagumo_eq(k, c, w1, w2, b, d, h, g, v, z, I):
    return [ k *(- c * z *( z -w1 ) *( z -w2) - b* v - d * v * z + I),
            k * h * (z - g * v)]


def cardiac_ode(R, t, parameters):
    x1, y1, x2, y2, x3, y3, x4, y4 = R[:8]
    zs = R[8::2]
    vs = R[9::2]

    p = SimpleNamespace(**{key: float(val) for key, val in parameters.items()})

    dRdt = [y1, -p.a1 * y1 * (x1 - p.u11) * (x1 - p.u12) - p.f1 * x1 * (x1 + p.d1) * (x1 + p.e1),
            y2,
            -p.a2 * y2 * (x2 - p.u21) * (x2 - p.u22) - p.f2 * x2 * (x2 + p.d2) * (x2 + p.e2) + p.k_sa_av * (x1 - x2),
            y3,
            -p.a3 * y3 * (x3 - p.u31) * (x3 - p.u32) - p.f3 * x3 * (x3 + p.d3) * (x3 + p.e3) + p.k_av_rb * (x2 - x3),
            y4,
            -p.a3 * y4 * (x4 - p.u31) * (x4 - p.u32) - p.f3 * x4 * (x4 + p.d3) * (x4 + p.e3) + p.k_av_lb * (x2 - x4)]

    # The influence of the second terms in the lines of (20) is negligeable in
    # physiological, so we neglect them here and rewrite in a simpler form.
    y3_hp = y3 + y4
    k_vars = ['k_at_de', 'k_at_re', 'k_vn_de', 'k_vn_re']
    Is = get_fhn_currents(y1, y3_hp, *[parameters[k_var] for k_var in k_vars])
    for i, I in enumerate(Is):
        dRdt += get_fitzhugo_nagumo_eq(parameters[f'k{i + 1}'],
                                       parameters[f'c{i + 1}'],
                                       parameters[f'w{i + 1}1'],
                                       parameters[f'w{i + 1}2'],
                                       parameters[f'b{i + 1}'],
                                       parameters[f'dm{i + 1}'],
                                       parameters[f'h{i + 1}'],
                                       parameters[f'g{i + 1}'],
                                       vs[i], zs[i], I)
    return dRdt


def cardiac_fwd_model(parameters,
                      R_0=None,  # Initial conditions
                      t=None,  # argument t = None means it can be given any value
                      noise=0):
    if R_0 is None:
        R_0 = (0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0,
               0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0)

    if t is None:
        t = np.linspace(0, 10, 1001)

    ret_val = odeint(cardiac_ode, R_0, t, args=(parameters,))

    ret_val += noise * (np.random.random(ret_val.shape) - 0.5)
    return ret_val