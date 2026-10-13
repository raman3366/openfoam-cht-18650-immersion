#!/usr/bin/env python3
"""
Property tables for the dielectric coolant: a model liquid of the synthetic-ester
type, defined by the anchor points below (the article states which of them have a
traced source).

Anchors used
------------
  rho(20 C)   = 970 kg/m3        cubic expansion  ~7.3e-4 1/K
  nu(40 C)    = 26.0 mm2/s       nu(100 C) = 5.3 mm2/s     -> Arrhenius fit
  k(20 C)     = 0.150 W/m/K      dk/dT = -6.5e-5 W/m/K/K
  cp(20 C)    = 1880 J/kg/K      dcp/dT = +3.6 J/kg/K/K

The Arrhenius fit nu = A exp(B/T) is the important one: viscosity falls ~8x
between 35 C and 110 C, which directly sets the natural-convection strength.
Constant-mu would be a poor approximation over this range, hence tabulated
transport (OpenFOAM 'tabulated' + 'icoTabulated' + 'hTabulated').
"""
import math

T0 = 293.15
RHO0, DRHO_DT = 970.0, -0.708      # kg/m3, kg/m3/K   (beta ~ 7.3e-4 1/K at 20 C)
K0,   DK_DT   = 0.150, -6.5e-5     # W/m/K
CP0,  DCP_DT  = 1880.0, 3.6        # J/kg/K

# Arrhenius nu = A exp(B/T) through (313.15 K, 26.0 cSt) and (373.15 K, 5.3 cSt)
T1, N1 = 313.15, 26.0e-6
T2, N2 = 373.15, 5.30e-6
B = math.log(N1 / N2) / (1.0 / T1 - 1.0 / T2)
A = N1 / math.exp(B / T1)

rho = lambda T: RHO0 + DRHO_DT * (T - T0)
nu  = lambda T: A * math.exp(B / T)
mu  = lambda T: rho(T) * nu(T)
kap = lambda T: K0 + DK_DT * (T - T0)
cp  = lambda T: CP0 + DCP_DT * (T - T0)

TS = list(range(280, 431, 10))     # 6.85 C .. 156.85 C, brackets the whole run

def table(f, fmt="{:.6g}"):
    return "\n".join("            ({:d} {})".format(T, fmt.format(f(T))) for T in TS)

if __name__ == "__main__":
    print("# Arrhenius fit:  nu = {:.6e} * exp({:.2f}/T)   [m2/s]".format(A, B))
    print("#    T[K]   T[C]     rho      nu[cSt]     mu[Pa.s]     k        cp      Pr")
    for T in TS:
        Pr = mu(T) * cp(T) / kap(T)
        print("  {:6.1f} {:6.1f}  {:7.1f}  {:9.2f}  {:10.3e}  {:6.4f}  {:6.0f}  {:6.1f}"
              .format(T, T - 273.15, rho(T), nu(T) * 1e6, mu(T), kap(T), cp(T), Pr))
    print()
    print("--- rho table ---");   print(table(rho, "{:.4f}"))
    print("--- mu table ---");    print(table(mu,  "{:.6e}"))
    print("--- kappa table ---"); print(table(kap, "{:.6f}"))
    print("--- Cp table ---");    print(table(cp,  "{:.2f}"))
