import re
from functools import cached_property, reduce
from typing import List, Tuple

import numpy as np
import sympy as sy
from IPython.display import Math, display
from pydantic import BaseModel
from scipy.signal import tf2zpk
from sympy import Add, Mul, Pow, Symbol, exp, latex, pi

from audio_utils import DEFAULT_SAMPLE_RATE


def replace_latex_powers(latex_str):
    return re.sub(r"\\left\(z\^\{-1\}\\right\)\^\{(\d+)\}", r"z^{-\1}", latex_str)


def evalf_poly(expr, var):
    return reduce(
        lambda x, y: Add(x, y, evaluate=True),
        [
            c.evalf() * var**ix
            for ix, c in enumerate(reversed(expr.as_poly(var).all_coeffs()))
        ],
    )


def poly_coeffs(expr, var):
    return expr.as_poly(var).all_coeffs()


Z_INV = sy.Symbol("z^-1")
Z = sy.Symbol("z")


class FiniteImpulseResponse(BaseModel):
    response: Tuple[float, ...]

    def get_z_transform(self):
        """Get the z-transform of the FIR filter."""
        return Add(*[v * Z**-i for i, v in enumerate(self.response)], evaluate=False)

    def get_z_transform_latex(self):
        resp = latex(self.get_z_transform().subs(Z, Z**-1).simplify().evalf())

        # split and reverse on the + sign
        resp = " + ".join(resp.split(" + ")[::-1])

        # replace z with z^{-1}, z^{2} with z^{-2}, etc.
        resp = re.sub(r"z([^\^])", r"z^{-1}\1", resp)
        resp = re.sub(r"z\^{(\d+)}", r"z^{-\1}", resp)

        return resp

    def as_faust_fir(self):
        """
        See: https://faustlibraries.grame.fr/libs/filters/#fifir
        """
        return f"(fi.fir(({','.join(map(str, self.response))}))"

    def as_faust_conv(self):
        """
        See: https://faustlibraries.grame.fr/libs/filters/#ficonv
        """
        return f"(fi.conv(({','.join(map(str, self.response))}))"


class InfiniteImpulseResponse(BaseModel):
    # Tuples b and a of indeterminate length
    b: Tuple[float, ...]
    a: Tuple[float, ...]

    def __init__(self, normalize=True, **data):
        super().__init__(**data)

        if normalize:
            # force a[0] to be 1
            self.b = tuple(np.array(self.b) / self.a[0])
            self.a = tuple(np.array(self.a) / self.a[0])

        self._fir_numerator = FiniteImpulseResponse(response=self.b)
        self._fir_denominator = FiniteImpulseResponse(response=self.a)

    def get_numerator_z_transform(self):
        return self._fir_numerator.get_z_transform()

    def get_denominator_z_transform(self):
        return self._fir_denominator.get_z_transform()

    def get_numerator_z_transform_latex(self):
        return self._fir_numerator.get_z_transform_latex()

    def get_denominator_z_transform_latex(self):
        return self._fir_denominator.get_z_transform_latex()

    def get_transfer_function(self):
        return self.get_numerator_z_transform() / self.get_denominator_z_transform()

    def get_transfer_function_latex(self):
        return f"\\dfrac{{{self.get_numerator_z_transform_latex()}}}{{{self.get_denominator_z_transform_latex()}}}"

    def show_transfer_function(self):
        display(Math(self.get_transfer_function_latex()))

    def is_stable(self):
        zeros, poles, _ = tf2zpk(self.b, self.a)
        return bool(np.all(np.abs(poles) < 1))

    def as_faust_iir(self):
        """
        See: https://faustlibraries.grame.fr/libs/filters/
        """

        if not (self.a[0] - 1) < 1e-10:
            raise ValueError("a[0] must be 1")

        return f"(fi.iir(({','.join(map(str, self.b))}),({','.join(map(str, self.a[1:]))}))"


class BiQuad(InfiniteImpulseResponse):
    b: Tuple[float, float, float]
    a: Tuple[float, float, float]

    @classmethod
    def from_coefficients(cls, b0, b1, b2, a0, a1, a2):
        return cls(b=(b0, b1, b2), a=(a0, a1, a2))

    def as_faust_tf21t(self):
        """
        See: https://faustlibraries.grame.fr/libs/filters/#fitf21t
        _ : tf21(b0,b1,b2,a1,a2) : _
        _ : tf22(b0,b1,b2,a1,a2) : _
        _ : tf22t(b0,b1,b2,a1,a2) : _
        _ : tf21t(b0,b1,b2,a1,a2) : _
        """
        return f"(fi.tf21t({','.join(map(str, self.b)) + ',' + ','.join(map(str, self.a[1:]))}))"

    def as_faust_tf22(self):  # don't use
        return f"(fi.tf22({','.join(map(str, self.b)) + ',' + ','.join(map(str, self.a[1:]))}))"

    def as_faust_tf22t(self):  # use
        return f"(fi.tf22t({','.join(map(str, self.b)) + ',' + ','.join(map(str, self.a[1:]))}))"

    def as_faust_tf21(self):  # use
        return f"(fi.tf21({','.join(map(str, self.b)) + ',' + ','.join(map(str, self.a[1:]))}))"


class CascadedBiquads(BaseModel):
    biquads: List[BiQuad]

    def as_faust(self):
        if len(self.biquads) == 0:
            return "(_:_)"
        return "(" + ":".join([bq.as_faust_tf21() for bq in self.biquads]) + ")"


class ParallelBiquads(BaseModel):
    biquads: List[BiQuad]

    def as_faust(self):
        if len(self.biquads) == 0:
            return "(_:_)"
        return (
            "( _ <: ("
            + ",".join([bq.as_faust_tf21() for bq in self.biquads])
            + ") :> _)"
        )


class Resonator:
    def __init__(self, f_k, tau, A, f_s=DEFAULT_SAMPLE_RATE):
        self.f_k = Symbol("f_k")
        self.f_s = Symbol("f_s")
        self.tau = Symbol("tau")
        self.A = Symbol("A")
        self.z_inv = Symbol("z^-1")

        self.values = {self.f_k: f_k, self.f_s: f_s, self.tau: tau, self.A: A}
        self.p_k = exp(2j * pi * f_k / f_s) * exp(-1 / (tau * f_s))
        self.a_2_k = abs(self.p_k) ** 2
        self.a_1_k = -2 * sy.re(self.p_k)
        self.b_k = A / f_s * sy.im(self.p_k)
        self.H_res_numerator = self.b_k * self.z_inv
        self.H_res_denominator = (
            1 + self.a_1_k * self.z_inv + self.a_2_k * self.z_inv**2
        )

        # self.verify_resonator()

    def __repr__(self):
        return f"Resonator(f_k={self.f_k.subs(self.values)}, tau={self.tau.subs(self.values)}, A={self.A.subs(self.values)})"

    def verify_resonator(self):
        # check for stability
        if not self.check_stability():
            raise RuntimeError("The resonator is not stable")
        if (
            not (self.estimate_resonant_frequency() - self.f_k.subs(self.values))
            / self.f_k.subs(self.values)
            < 0.01
        ):
            raise RuntimeError(
                "The resonator is not resonating at the desired frequency"
            )

    def transfer_function(self):
        return Mul(
            (self.H_res_numerator),
            Pow((self.H_res_denominator), -1, evaluate=True),
            evaluate=False,
        )

    def evalf_transfer_function(self):
        return evalf_poly(self.H_res_numerator, self.z_inv), evalf_poly(
            self.H_res_denominator, self.z_inv
        )

    def latex_transfer_function(self):
        return replace_latex_powers(latex(self.transfer_function()))

    def display_transfer_function(self):
        evalf_num = evalf_poly(self.H_res_numerator.subs(self.values), self.z_inv)
        evalf_denom = evalf_poly(self.H_res_denominator.subs(self.values), self.z_inv)

        print("transfer function:")
        display(
            Math(
                r"H_\text{res} = "
                + replace_latex_powers(
                    latex(
                        Mul(
                            (evalf_num),
                            Pow((evalf_denom), -1, evaluate=True),
                            evaluate=False,
                        )
                    )
                )
            )
        )

    @cached_property
    def b(self):
        b = np.array(poly_coeffs(self.H_res_numerator, self.z_inv)[::-1]).astype(
            np.float64
        )

        # pad with zeros to make it 3 coefficients
        b = np.pad(b, (0, 3 - len(b)), "constant")
        return b

    @cached_property
    def a(self):
        a = np.array(poly_coeffs(self.H_res_denominator, self.z_inv)[::-1]).astype(
            np.float64
        )
        # pad with zeros to make it 3 coefficients
        a = np.pad(a, (0, 3 - len(a)), "constant")
        return a

    def estimate_resonant_frequency(self):
        import numpy as np
        from scipy.signal import freqz

        w, h = freqz(self.b, self.a, worN=16384)
        # find the frequency with the maximum amplitude
        max_freq = w[np.argmax(abs(h))]

        # normalize the frequency to the range [0, f_s/2] from [0, pi]
        return max_freq * self.f_s.subs(self.values) / (2 * np.pi)

    def check_stability(self):
        import numpy as np
        from scipy.signal import tf2zpk

        b, a = self.get_b_and_a()
        zeros, poles, _ = tf2zpk(b, a)
        return np.all(np.abs(poles) < 1)

    def as_faust_tf21t(self):
        return f"(fi.tf21t({self.b[0]}, {self.b[1]}, {self.b[2]}, {self.a[1]}, {self.a[2]}))"
