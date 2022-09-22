
# Copied and modified from https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
#  # the cardiac equations are given by 2 ode's

# ode for x1-SV

from types import SimpleNamespace
import numpy as np
from scipy.integrate import odeint
from scipy.special import expit
from pathlib import Path
import json

def threshold_fct(x, c=10):
    return expit(c*x)


class QuirozJuarezModel:
    nominal_parameters = {'h': 2.164,
                          'c': 1.35,
                          'b': 4,
                          'g': 7,  # Time scaling
                          'to': 0,  # Time offset
                          'a1': -0.024,
                          'a2': 0.0216,
                          'a3': -0.0012,
                          'a4': 0.12}

    @property
    def parameter_names(self):
        return list(self.nominal_parameters.keys())

    def __init__(self, parameters=None):

        self.set_parameters(self.nominal_parameters)
        if parameters is not None:
            self.set_parameters(parameters)

    @property
    def parameters(self):
        return {param_name: getattr(self, param_name) for param_name in self.parameter_names}

    def set_parameters(self, parameters):
        for param_name, param_val in parameters.items():
            self.parameter_names.append(param_name)
            setattr(self, param_name, float(param_val))

    def _cardiac_ode(self, R, t):

        x1, x2, x3, x4 = R

        p = self
        dRdt = [p.g * (x1 - x2 - p.c * x1 * x2 - x1 * x2 ** 2),
                p.g * (p.h * x1 - 3 * x2 + p.c * x1 * x2 + x1 * x2 ** 2 + p.b * (x4 - x2)),
                p.g * (x3 - x4 - p.c * x3 * x4 - x3 * x4 ** 2),
                p.g * (p.h * x3 - 3 * x4 + p.c * x3 * x4 + x3 * x4 ** 2 + 2 * p.b * (x2 - x4))]

        return dRdt

    def cardiac_fwd_model(self, parameters=None,
                          R_0=None,  # Initial conditions
                          t=None,  # argument t = None means it can be given any value
                          noise=0):

        assert (self.to >= -1)
        assert (self.to <= 1)

        if parameters is not None:
            self.set_parameters(parameters)

        if R_0 is None:
            R_0 = (0.1, 0.1, 0.1, 0.1)

        if t is None:
            t = np.linspace(0, 10, 1001)

        # To offset the signal
        t = np.concatenate((np.linspace(-1, self.to, 201) - self.to + t[0], t))
        ret_val = odeint(self._cardiac_ode, R_0, t)[201:]

        ret_val += noise * (np.random.random(ret_val.shape) - 0.5)
        return ret_val

    def ecg_fwd_model(self, **kwargs):
        x1, x2, x3, x4 = self.cardiac_fwd_model(**kwargs).T
        return self.a1 * x1 + self.a2 * x2 + self.a3 * x3 + self.a4 * x4


class CardarilliModel:

    def __init__(self, parameters=None):

        self.parameter_names = []

        self.load_nominal_parameters()
        if parameters is not None:
            self.set_parameters(parameters)

    @property
    def parameters(self):
        return {param_name: getattr(self, param_name) for param_name in self.parameter_names}

    def load_nominal_parameters(self):
        json_path = Path(__file__).parent.parent / "data" / "parameters_nominal_cardarilli.json"
        self.set_parameters(json.load(json_path.open()))


    def set_parameters(self, parameters):
        for param_name, param_val in parameters.items():
            self.parameter_names.append(param_name)
            setattr(self, param_name, float(param_val))


    def get_fhn_currents(self, y1, y3_hp):
        # Ensuring differentiability by replacing the discontinuity
        # by a continuous function.

        # I_at_de = np.where(k_I[0]*y1, y1> 0, 0)
        I_at_de = self.k_at_de *y1 *threshold_fct(y1)

        # I_at_re = np.where(-k_I[1]*y1, y1<= 0, 0)
        I_at_re = -self.k_at_re *y1 *threshold_fct(-y1)

        # I_vn_de = np.where(k_I[2]*0.5*y3_hp, y3_hp > 0, 0)
        I_vn_de = self.k_vn_de *0.5 *y3_hp *threshold_fct(y3_hp)

        # I_vn_re = np.where(-k_I[3]*0.5*y3_hp, y3_hp <= 0, 0)
        I_vn_re = -self.k_vn_re *0.5 *y3_hp *threshold_fct(-y3_hp)

        return [I_at_de, I_at_re, I_vn_de, I_vn_re]


    def _cardiac_ode(self, R, t):

        def get_fitzhugo_nagumo_eq(k, c, w1, w2, b, d, h, g, v, z, I):
            return [k * (- c * z * (z - w1) * (z - w2) - b * v - d * v * z + I),
                    k * h * (z - g * v)]

        x1, y1, x2, y2, x3, y3, x4, y4 = R[:8]
        zs = R[8::2]
        vs = R[9::2]

        p = self
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
        for i, I in enumerate(self.get_fhn_currents(y1, y3_hp)):
            dRdt += get_fitzhugo_nagumo_eq(getattr(self, f'k{i + 1}'),
                                           getattr(self, f'c{i + 1}'),
                                           getattr(self, f'w{i + 1}1'),
                                           getattr(self, f'w{i + 1}2'),
                                           getattr(self, f'b{i + 1}'),
                                           getattr(self, f'dm{i + 1}'),
                                           getattr(self, f'h{i + 1}'),
                                           getattr(self, f'g{i + 1}'),
                                           vs[i], zs[i], I)
        return dRdt


    def cardiac_fwd_model(self, parameters=None,
                          R_0=None,  # Initial conditions
                          t=None,  # argument t = None means it can be given any value
                          noise=0):

        if parameters is not None:
            self.set_parameters(parameters)

        if R_0 is None:
            R_0 = (0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0,
                   0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0)

        if t is None:
            t = np.linspace(0, 10, 1001)

        ret_val = odeint(self._cardiac_ode, R_0, t)

        ret_val += noise * (np.random.random(ret_val.shape) - 0.5)
        return ret_val


    def ecg_fwd_model(self, k_r=2, z0=0.2, **kwargs):
        states = self.cardiac_fwd_model(**kwargs)
        return z0 + states[:, 8] - states[:, 10] + k_r*states[:, 12] + states[:, 14]
