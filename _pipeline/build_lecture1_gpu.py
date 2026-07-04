#!/usr/bin/env python
"""Assemble content/Lecture1.ipynb (unexecuted). Execute + render via build.py.

Lecture 1 -- Overview & a First Model Atmosphere. This is written as
self-contained graduate lecture notes: the stellar-spectrum forward problem,
the Planck function, LTE, optical depth, and a grey solar atmosphere built from
(Teff, log g).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture1.ipynb"

cells = []


def md(src):
    cells.append(new_markdown_cell(src))


def code(src):
    cells.append(new_code_cell(src))


md(r"""# Lecture 1 — Overview & a First Model Atmosphere

*Self-contained notes on the forward problem of stellar spectroscopy*

*Yuan-Sen Ting*

A stellar spectrum is not just a curve of brightness against wavelength. It is the record of photons being created, absorbed, scattered, and finally released from the outer layers of a star. The same spectrum contains several kinds of information at once. Its broad continuum scale is mainly thermal. Its absorption lines remember which atoms and ions are present, how their energy levels are populated, and how pressure and velocity fields shape the escaping radiation.

The theoretical task is to predict that spectrum from physical inputs. A synthetic spectrum is the forward prediction: given a star's **effective temperature** $T_{\rm eff}$, **surface gravity** $g$, and chemical composition, what flux should emerge at each wavelength?

The first object to build is not a line list or a transfer solver. It is an atmosphere: a run of temperature, pressure, and depth through the photosphere. The version built here is deliberately simple, but it already has the pieces every stellar-atmosphere calculation needs: a thermal radiation scale, a depth coordinate tied to photon escape, a temperature structure, and a pressure scale.""")


md(r"""## The Forward Problem

An observed stellar spectrum gives the flux received as a function of wavelength. Reading physical parameters from that spectrum is an inverse problem. The more controlled problem is the forward one:

$$
(T_{\rm eff},\ \log g,\ \mathrm{composition})
\quad\longrightarrow\quad
F_\lambda .
$$

Here $F_\lambda$ is the flux emerging from the stellar surface per unit wavelength. Flux is energy per unit area per unit time per wavelength interval, integrated over the outward directions. It is what a distant observer receives after the usual geometric dilution by distance.

The first input, $T_{\rm eff}$, is the **effective temperature**. It is defined by the total emergent flux:

$$
F = \sigma T_{\rm eff}^4,
$$

where $\sigma$ is the Stefan--Boltzmann constant. Thus $T_{\rm eff}$ is the temperature of a blackbody that emits the same total flux as the star. It is not the temperature at every depth. A photosphere has a temperature gradient: deeper layers are hotter, and photons at different wavelengths can escape from different depths.

The second input is $\log g$, the base-10 logarithm of the surface gravity in CGS units. Surface gravity is the downward acceleration felt by material near the stellar surface. A solar value $\log g=4.44$ means

$$
g \approx 10^{4.44}\ \mathrm{cm\,s^{-2}} .
$$

Gravity matters because gas pressure must support the weight of the material above each layer. At fixed temperature, a high-gravity atmosphere is more compressed than a low-gravity atmosphere.

The composition matters because atoms, ions, molecules, and free electrons determine the opacity. Opacity is what decides which photons can escape and which photons are absorbed or scattered before leaving the star.

It is useful to separate the calculation into two physical objects:

1. A **model atmosphere** gives temperature, pressure, density, and related thermodynamic quantities as functions of depth.
2. **Spectral synthesis** uses that atmosphere to compute wavelength-dependent opacity and solve how radiation escapes through it.

The model atmosphere is the physical stage on which the spectrum is formed. Without it, the phrase "a line forms higher than the continuum" has no meaning, because there is no depth scale and no temperature or pressure assigned to that depth.

The atmosphere below uses four simplifying assumptions.

**Static** means there is no time dependence and no bulk acceleration. The gas is allowed to have pressure and gravity, but it is not expanding, collapsing, or pulsating.

**Plane-parallel** means a small patch of the photosphere is treated as a flat slab. This is appropriate when the thickness of the line-forming layers is tiny compared with the stellar radius.

**Grey** means the opacity is treated as independent of wavelength. Real opacities vary strongly with wavelength; the grey approximation is a controlled way to get a first temperature-pressure structure before detailed line opacity is introduced.

**Local thermodynamic equilibrium** (LTE) means the material at each depth is described by a local temperature. The radiation field can stream outward and need not be a blackbody everywhere, but the matter locally has thermal level populations and thermal emission set by $T$.

![A synthetic spectrum is the last step of a chain: stellar parameters define an atmosphere, the atmosphere sets the material state and opacity, and radiative transfer gives the escaping flux.](resources/figures/s1_pipeline.png)""")


md(r"""## Numerical Conventions

The numerical examples use PyTorch so the same expressions can run on a GPU or on the CPU. Astropy supplies physical constants with units; after unit conversion, the calculation itself is ordinary tensor arithmetic.""")

code(r'''import numpy as np
import torch
import matplotlib.pyplot as plt
from astropy import constants as astro_constants
from astropy import units as astro_units

plt.rcParams.update({
    "figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
})

# Only execution machinery: the atmosphere equations below do not depend on this choice.
if torch.backends.mps.is_available():
    EXECUTION_DEVICE, FLOAT_TYPE = torch.device("mps"), torch.float32
elif torch.cuda.is_available():
    EXECUTION_DEVICE, FLOAT_TYPE = torch.device("cuda"), torch.float32
else:
    EXECUTION_DEVICE, FLOAT_TYPE = torch.device("cpu"), torch.float64

def host_float(tensor_scalar):
    """Convert a tensor scalar to a Python float for printing or plot annotation."""
    return float(torch.as_tensor(tensor_scalar).detach().cpu())

def to_device(value, *, dtype=None):
    """Create a tensor on the execution device."""
    return torch.as_tensor(value, dtype=dtype or FLOAT_TYPE, device=EXECUTION_DEVICE)

def display_array(tensor):
    """Move a completed tensor result to CPU/NumPy for plotting or table formatting."""
    return tensor.detach().cpu().to(torch.float64).numpy()''')


md(r"""## Units and Constants

Stellar-atmosphere calculations are traditionally written in **Gaussian CGS** units: centimetres, grams, and seconds. The choice is conventional but useful: opacities are commonly tabulated in $\mathrm{cm^2\,g^{-1}}$, pressures in $\mathrm{dyn\,cm^{-2}}$, and wavelengths in centimetres when they appear inside formulae.

Three constants set the radiation scale:

| symbol | name | value (CGS) | role |
|---|---|---|---|
| $h$ | Planck constant | $6.62607015\times10^{-27}\ \mathrm{erg\,s}$ | photon energy, $E=h\nu$ |
| $c$ | speed of light | $2.99792458\times10^{10}\ \mathrm{cm\,s^{-1}}$ | wavelength-frequency relation, $\lambda\nu=c$ |
| $k$ | Boltzmann constant | $1.380649\times10^{-16}\ \mathrm{erg\,K^{-1}}$ | thermal energy scale, $kT$ |

The first useful scale is the energy of an optical photon. At $500\,\mathrm{nm}$, the photon energy is a few electron-volts, while $kT$ in the solar photosphere is about half an electron-volt. Optical photons therefore probe the same energy range as many atomic excitation and ionization processes. This is why optical stellar spectra contain so much chemical information.""")

code(r'''planck_constant = to_device(astro_constants.h.cgs.value)       # h [erg s]
light_speed = to_device(astro_constants.c.cgs.value)           # c [cm s^-1]
boltzmann_constant = to_device(astro_constants.k_B.cgs.value)  # k [erg K^-1]
erg_per_ev = to_device(astro_units.eV.to(astro_units.erg))

wavelength_500_cm = to_device(500e-7)
solar_temperature = to_device(5770.0)

photon_energy_500 = planck_constant * light_speed / wavelength_500_cm
thermal_energy_sun = boltzmann_constant * solar_temperature

planck_prefactor_cgs = 2.0 * astro_constants.h.cgs.value / astro_constants.c.cgs.value**2
frequency_scale_hz = to_device(1.0e15)
planck_nu_prefactor = to_device(planck_prefactor_cgs * 1.0e45)
radiation_pressure_coefficient = to_device(
    4.0 * astro_constants.sigma_sb.cgs.value / (3.0 * astro_constants.c.cgs.value)
)

print(f"E(500 nm) = h*c/lambda = {host_float(photon_energy_500):.3e} erg = {host_float(photon_energy_500/erg_per_ev):.3f} eV")
print(f"kT at 5770 K          = {host_float(thermal_energy_sun):.3e} erg = {host_float(thermal_energy_sun/erg_per_ev):.3f} eV")
print(f"Planck prefactor 2h/c^2 = {planck_prefactor_cgs:.6e}")''')


md(r"""## Thermal Radiation: The Planck Function

A stellar atmosphere is not a solid surface with a single temperature. Radiation is emitted and absorbed throughout a layer of gas. The first local quantity we need is therefore the radiation produced by matter at one temperature.

A gas in thermodynamic equilibrium at temperature $T$ emits a blackbody spectrum. The relevant quantity is **specific intensity**, usually written $I_\nu$ or $I_\lambda$. Specific intensity measures radiation per unit area, per unit time, per unit frequency or wavelength interval, per unit solid angle. It keeps track of direction, which later matters because photons moving vertically and photons moving at a slant pass through different amounts of material.

The blackbody specific intensity per unit frequency is the **Planck function**:

$$
B_\nu(T)
=
\frac{2h\nu^3}{c^2}\,
\frac{1}{e^{h\nu/kT}-1}
\qquad
[\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}].
$$

The factors have simple meanings. The $2\nu^2/c^2$ part counts electromagnetic modes per frequency interval and solid angle. The extra factor $h\nu$ is the energy per photon. The denominator $e^{h\nu/kT}-1$ is the Bose-Einstein occupation factor for photons in thermal equilibrium. The dimensionless ratio

$$
x \equiv \frac{h\nu}{kT}
$$

compares photon energy to thermal energy. When $x$ is small, many photons occupy the mode. When $x$ is large, the mode is exponentially hard to populate.

Spectra are often plotted against wavelength. The wavelength form is not obtained by simply replacing $\nu$ with $c/\lambda$, because a frequency bin and a wavelength bin have different widths. Conservation of energy in the bin requires

$$
B_\lambda\,|d\lambda| = B_\nu\,|d\nu|.
$$

Since $\nu=c/\lambda$, we have $|d\nu/d\lambda|=c/\lambda^2$, and therefore

$$
B_\lambda(T)
=
B_\nu(T)\frac{c}{\lambda^2}
=
\frac{2hc^2}{\lambda^5}\,
\frac{1}{e^{hc/\lambda kT}-1}.
$$

Here $\lambda$ is measured in centimetres if CGS units are used. This distinction between $B_\nu$ and $B_\lambda$ is a common source of mistakes: they describe the same radiation, but per different interval.

Two limits are worth keeping in mind. In the Rayleigh--Jeans limit, $h\nu\ll kT$, the exponential can be expanded and

$$
B_\nu \approx \frac{2\nu^2 kT}{c^2}.
$$

In the Wien limit, $h\nu\gg kT$,

$$
B_\nu \approx \frac{2h\nu^3}{c^2}e^{-h\nu/kT}.
$$

Optical photons in a solar photosphere lie between these limits. The energy scale is high enough that the intensity changes rapidly with temperature, which is one reason optical spectra are powerful temperature diagnostics.""")


md(r"""For numerical work it is better to avoid the literal denominator $e^{h\nu/kT}-1$, because $e^x$ can become very large in the Wien tail. The physics is unchanged; only the algebraic form changes. With

$$
x \equiv \frac{h\nu}{kT}.
$$

Multiplying numerator and denominator by $e^{-x}$ gives

$$
\frac{1}{e^x-1}
=
\frac{e^{-x}}{1-e^{-x}},
$$

which uses an exponential that is always $\leq1$. With frequency measured as

$$
\nu_{15}\equiv\frac{\nu}{10^{15}\,\mathrm{Hz}},
$$

a convenient implementation is

$$
B_\nu
=
\left[\frac{2h}{c^2}(10^{15}\,\mathrm{Hz})^3\right]\,
\nu_{15}^3\,
\frac{e^{-x}}{1-e^{-x}}.
$$

The bracketed prefactor is about $1.4745\times10^{-2}$ in CGS units. The implementation keeps this prefactor explicit and lets PyTorch carry out the tensor calculation on the selected device.""")

code(r'''def planck_nu(frequency_hz, temperature_K):
    """Planck B_nu(T) in CGS [erg s^-1 cm^-2 Hz^-1 sr^-1].

    frequency_hz : photon frequency nu [Hz]
    temperature_K: gas temperature T   [K]
    """
    frequency = to_device(frequency_hz)
    temperature = to_device(temperature_K)

    exponent = planck_constant * frequency / (boltzmann_constant * temperature)
    exp_minus_exponent = torch.exp(-exponent)
    occupation_factor = exp_minus_exponent / (1.0 - exp_minus_exponent)

    scaled_frequency = frequency / frequency_scale_hz
    return planck_nu_prefactor * scaled_frequency**3 * occupation_factor''')


md(r"""The figure below converts $B_\nu$ to $B_\lambda$ and evaluates the Planck function for three temperatures. The hotter curves are brighter and peak at shorter wavelength. This is the continuum-level origin of the familiar color sequence from cool red stars to hot blue stars.

A solar-temperature blackbody peaks near the visible band, so a $500$--$510\,\mathrm{nm}$ window lies close to the maximum of the solar continuum. That makes the window a useful place to learn the line physics without first fighting very small continuum fluxes.""")

code(r'''wavelength_nm = torch.linspace(200.0, 2000.0, 400, dtype=FLOAT_TYPE, device=EXECUTION_DEVICE)
wavelength_cm = wavelength_nm * to_device(1.0e-7)

temperatures_for_plot = to_device([4500.0, 5770.0, 7500.0]).reshape(3, 1)
frequency_grid = light_speed / wavelength_cm.reshape(1, -1)

planck_per_frequency = planck_nu(frequency_grid, temperatures_for_plot)
planck_per_wavelength = planck_per_frequency * light_speed / wavelength_cm.reshape(1, -1)**2

plt.plot(
    display_array(wavelength_nm),
    display_array(planck_per_wavelength.T),
)
plt.axvspan(500, 510, color="0.5", alpha=0.25, label="our window")
plt.xlabel("wavelength  [nm]")
plt.ylabel(r"$B_\lambda(T)$  [erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$ sr$^{-1}$]")
plt.title("The Planck function from the near-UV to the near-IR")
plt.legend(["T = 4500 K", "T = 5770 K", "T = 7500 K", "our window"])
plt.tight_layout()
plt.show()''')


md(r"""At a fixed optical wavelength, the same temperature sensitivity appears as a steep rise in $B_\nu$. At $505\,\mathrm{nm}$, the three temperatures used in the plot give:""")

code(r'''frequency_505 = light_speed / to_device(505e-7)
temperatures_for_check = to_device([4500.0, 5770.0, 7500.0])
planck_505 = planck_nu(frequency_505, temperatures_for_check)

planck_table = display_array(torch.stack((temperatures_for_check, planck_505), dim=1))
print("columns: temperature [K],  B_nu(505 nm)")
print(np.array2string(planck_table, precision=4))''')


md(r"""## Optical Depth and the Photosphere

The Planck function tells us what thermal radiation matter would emit locally at temperature $T$. It does not tell us whether that radiation reaches space. A photon created deep in a star can be absorbed or scattered many times before it escapes, while a photon created near the top may leave almost immediately.

This means the natural depth coordinate is not geometric height by itself. A photon does not mainly care how many kilometres it travels. It cares how many chances it has to interact with matter before reaching free space. If the opacity is large, a physically thin layer can be opaque. If the opacity is small, a physically thick layer can be nearly transparent.

The **mass extinction coefficient** $\kappa_\lambda$ has units $\mathrm{cm^2\,g^{-1}}$. It is the effective blocking area per gram of material at wavelength $\lambda$. The subscript $\lambda$ is a reminder that real opacity depends on wavelength. In this first atmosphere, $\kappa_\lambda$ is treated as a material property: it tells us how strongly one gram of gas removes light from a beam. Computing $\kappa_\lambda$ from atoms, ions, molecules, free electrons, and spectral lines is a separate physics problem; here we only need what opacity does once it is known.

Multiplying by the mass density $\rho$ gives $\kappa_\lambda\rho$ in $\mathrm{cm^{-1}}$, an extinction probability per unit length. Its inverse is the photon mean free path,

$$
\ell_\lambda = \frac{1}{\kappa_\lambda\rho}.
$$

**Optical depth** is distance measured in mean-free-path units. A change $d\tau_\lambda=1$ means roughly one interaction length. Let $z$ increase outward. Looking vertically through the atmosphere,

$$
d\tau_\lambda = -\kappa_\lambda \rho\,dz .
$$

Here **extinction** means removal of photons from a beam by absorption plus scattering. The minus sign says that $\tau_\lambda$ increases inward: the deeper the photon starts, the more material lies above it. A photon born at $\tau_\lambda\gg1$ is many mean free paths from escape. A photon born at $\tau_\lambda\ll1$ is already in optically thin material.

The exponential attenuation law follows directly from this definition. Along a short path segment, the fractional loss of beam intensity is proportional to the optical-depth step:

$$
dI_\lambda = -I_\lambda\,d\tau_\lambda .
$$

Dividing by $I_\lambda$ and integrating from $\tau=0$ to $\tau$ gives

$$
\int_{I_0}^{I}\frac{dI_\lambda}{I_\lambda}
=
-\int_0^\tau d\tau_\lambda,
\qquad\Rightarrow\qquad
\ln\frac{I}{I_0}=-\tau .
$$

Therefore a beam that only loses photons and gains no new emission has

$$
I = I_0 e^{-\tau}.
$$

This attenuation law gives the practical meaning of optical depth: $\tau\ll1$ is transparent, $\tau\sim1$ is the transition between transparent and opaque, and $\tau\gg1$ is opaque.""")

code(r'''optical_depth_examples = to_device([0.01, 0.1, 1.0, 3.0, 10.0])
survival_fraction = torch.exp(-optical_depth_examples)

attenuation_table = display_array(torch.stack((optical_depth_examples, survival_fraction), dim=1))
print("columns: optical depth tau,  transmitted fraction exp(-tau)")
print(np.array2string(attenuation_table, precision=4))''')


md(r"""The table gives the basic intuition. At $\tau=0.1$, most photons survive. At $\tau=1$, only $e^{-1}\simeq0.37$ of a purely attenuated beam remains. At $\tau=10$, direct escape is essentially impossible. The visible photosphere is therefore a layer, not a solid surface: it is the transition region where the optical depth to space is around unity.

It is often cleaner to replace geometric height with **column mass**,

$$
m(z) = \int_z^\infty \rho(z')\,dz',
$$

the mass above one square centimetre of surface. Since $dm=-\rho\,dz$,

$$
d\tau_\lambda = \kappa_\lambda\,dm.
$$

This is the bridge between radiative transfer and hydrostatic structure. Optical depth says how far photons are from escape; column mass says how much weight lies above a layer. The subscript matters: this equation defines a different $\tau_\lambda$ for each wavelength because $\kappa_\lambda$ is generally different at each wavelength. Atmosphere tables are usually organized by one of these coordinates because both are more physically useful than geometric height.

There is also a more precise reason that optical depth of order unity appears in photosphere arguments. When material emits as well as absorbs, the emergent intensity is a weighted average of local emission along the ray. Material at very large optical depth is exponentially hidden, while material at very small optical depth contributes little because there is not much of it. For a ray with direction cosine $\mu=\cos\theta$, the main contribution comes from $\tau_\lambda\sim\mu$.

The observed flux is an angular average over outward rays, so the precise numerical depth depends on the angular boundary model. The robust physical point is that the photosphere is located around optical depth unity, not at a geometric surface. Different wavelengths and line strengths sample different depths because $\kappa_\lambda$ changes with wavelength. A strong line core has large opacity, reaches $\tau_\lambda\sim1$ higher in the atmosphere, and therefore samples cooler gas. A weak line wing or continuum point reaches $\tau_\lambda\sim1$ deeper down.

![Optical depth: photons escape from layers where the overlying optical depth is of order unity.](resources/figures/s1_optical_depth.png)""")


md(r"""## Local Thermodynamic Equilibrium

Optical depth tells us how hard it is for radiation to escape. We also need to know what the gas contributes to the radiation field while a beam passes through it.

The **source function** $S_\lambda$ is the local emission divided by the local extinction, with the same units as specific intensity. It is the intensity the material would like to impose on the radiation field locally. If the incoming intensity is smaller than $S_\lambda$, local emission tends to brighten the beam. If the incoming intensity is larger than $S_\lambda$, extinction tends to dim it.

In optical-depth form, the radiative-transfer equation along a ray is

$$
\mu\frac{dI_\lambda}{d\tau_\lambda}
=
I_\lambda-S_\lambda,
$$

where $\mu=\cos\theta$ is the direction cosine relative to the outward normal. The attenuation law came from the special case with no source term. Stellar spectra require the full balance: extinction removes photons from a beam, while the source function adds locally emitted photons.

The simplifying assumption is **local thermodynamic equilibrium**, or LTE. LTE means that, at each depth, collisions are frequent enough for the material state to be described by a local temperature $T$. The radiation field may stream outward and need not be a blackbody everywhere, but the matter locally behaves as if it were in thermodynamic equilibrium.

**Kirchhoff's law** says that, in thermodynamic equilibrium, material that absorbs efficiently at a wavelength must also emit efficiently at that wavelength. If this were not true, matter in equilibrium would radiatively heat up or cool down on its own, contradicting equilibrium.

Applied locally in LTE, Kirchhoff's law makes the thermal source function equal to the Planck function at the local temperature:

$$
S_\lambda = B_\lambda(T).
$$

Here **true absorption** means photons are converted into internal energy of the gas, followed by thermal re-emission. Scattering is different: it redirects photons and can have its own source term. For this first atmosphere, the essential LTE statement is simple: once we know $T$ at a depth, we know the local thermal emission scale there.""")


md(r"""## A Grey Model Atmosphere

A model atmosphere assigns physical conditions to depth. For a first calculation, the central quantity is the temperature structure $T(\tau)$, because LTE makes the thermal source function equal to $B_\lambda[T(\tau)]$. Once $T$ is known, hydrostatic equilibrium supplies a corresponding pressure scale.

The simplest useful atmosphere is a **grey atmosphere**. Grey means that the opacity is treated as independent of wavelength. Then there is one optical-depth coordinate $\tau$ rather than a different $\tau_\lambda$ at every wavelength. This approximation cannot produce absorption lines, because lines require wavelength-dependent opacity. Its purpose is narrower and important: it gives a continuum pressure-temperature scaffold from only $T_{\rm eff}$ and $\log g$.

The atmosphere is in **radiative equilibrium**. This means each layer passes along the same total radiative flux. A layer may absorb and emit radiation, but after all wavelengths and directions are counted it has no net radiative heating or cooling. The outward flux is fixed by the effective temperature:

$$
F = \sigma T_{\rm eff}^4.
$$

The grey temperature law can be understood without solving the full angle-dependent transfer problem. The useful idea is to compress the radiation field into angular moments. This uses the same specific intensity introduced with the Planck function, but now integrated over all frequencies and written only as a function of ray direction: $I(\mu)$. This frequency-integrated intensity is called **bolometric** intensity. In a plane-parallel atmosphere let $\mu=\cos\theta$, where $\theta$ is measured from the outward normal. The three lowest moments are

$$
J=\frac{1}{2}\int_{-1}^{1} I(\mu)\,d\mu,
\qquad
H=\frac{1}{2}\int_{-1}^{1} \mu I(\mu)\,d\mu,
\qquad
K=\frac{1}{2}\int_{-1}^{1} \mu^2 I(\mu)\,d\mu.
$$

Here $J$ is the angle-averaged intensity, $H$ is the flux moment, and $K$ is the radiation-pressure moment. The physical flux is

$$
F=4\pi H.
$$

In grey LTE, radiative equilibrium connects the local radiation field to the local thermal emission:

$$
J = B(T)=\frac{\sigma}{\pi}T^4.
$$

The total flux is constant with depth and is fixed by the effective temperature, so

$$
H=\frac{F}{4\pi}=\frac{\sigma T_{\rm eff}^4}{4\pi}.
$$

One more relation is needed because the moments contain $J$, $H$, and $K$. A **closure** is an approximation that relates the higher moment to the lower ones so the system can be solved. The Eddington approximation uses

$$
K\simeq \frac{J}{3}.
$$

This relation is exact for an isotropic radiation field, because the angular average of $\mu^2$ is $1/3$. It is not exact at the surface, where radiation streams preferentially outward, but it is the simplest useful closure for a grey atmosphere.

The moment equation for the flux then gives the compact result

$$
\frac{1}{3}\frac{dJ}{d\tau}=H.
$$

Substituting $J=(\sigma/\pi)T^4$ and $H=\sigma T_{\rm eff}^4/(4\pi)$ gives

$$
\frac{dT^4}{d\tau}=\frac{3}{4}T_{\rm eff}^4.
$$

This is the important physical result: in a grey radiative-equilibrium atmosphere, $T^4$ increases linearly with optical depth. Integrating gives

$$
T^4(\tau)
=
\frac{3}{4}T_{\rm eff}^4(\tau+C),
$$

where $C$ is a boundary constant. The slope is set by radiative equilibrium; the additive constant is set by the surface boundary condition. The simplest Eddington boundary gives $C=2/3$, so

$$
T^4(\tau)
=
\frac{3}{4}T_{\rm eff}^4\left(\tau+\frac{2}{3}\right).
$$

This makes $T(\tau=2/3)=T_{\rm eff}$. That is the origin of the familiar statement that the visible photosphere is near optical depth $2/3$. It is not a material surface. It is the depth where the emergent continuum samples gas with roughly the effective temperature.

More detailed grey surface solutions replace the constant $C$ by a dimensionless **Hopf function** $q(\tau)$:

$$
T^4(\tau)
=
\frac{3}{4}T_{\rm eff}^4[\tau+q(\tau)].
$$

The Hopf function is not an extra stellar parameter. It is a compact way to encode the angular boundary correction near the surface while leaving the deeper radiative-equilibrium gradient intact. In deep layers $q(\tau)$ approaches an almost constant offset, so the temperature gradient remains the one derived above.

The moment derivation fixes the slope of $T^4$ with optical depth, but it does not force one particular surface correction. The constant $q=2/3$ is the simplest Eddington boundary choice. The numerical atmosphere below instead uses the smooth Kurucz/ATLAS grey-start fit

$$
\tau + q(\tau)
=
0.710+\tau-0.1331e^{-3.4488\tau}.
$$

Substituting this bracket into the grey law gives

$$
T(\tau)
=
T_{\rm eff}\,
\left[
\frac{3}{4}
\left(0.710+\tau-0.1331e^{-3.4488\tau}\right)
\right]^{1/4}.
$$

This is a modeling choice made for the grey starting atmosphere, not a new physical law. The exponential term changes only the surface layers: $q(0)=0.710-0.1331=0.5769$, so the top boundary is slightly cooler than the constant-$q$ Eddington form, while deep in the atmosphere the exponential dies away and $q(\tau)$ approaches $0.710$. The result keeps $T(\tau=2/3)$ very close to $T_{\rm eff}$ while giving a smoother surface behavior than a constant $q$.

We now need to turn the temperature law into a table of layers. The variable in the temperature law is an optical depth, so the grid should be spaced in optical depth rather than in geometric height.

**Rosseland optical depth.** This is not a new kind of optical depth. It is the same construction as $d\tau_\lambda=\kappa_\lambda\,dm$, but with the many wavelength-dependent opacities replaced by one representative opacity. In a real atmosphere, blue continuum, red continuum, weak line wings, and strong line cores all see different $\kappa_\lambda$. A model atmosphere still needs one common depth coordinate on which to tabulate $T$, pressure, and column mass. The standard choice is the **Rosseland optical depth** $\tau_{\rm Ross}$, defined by

$$
d\tau_{\rm Ross}=\kappa_{\rm Ross}\,dm,
$$

where $m$ is column mass and $\kappa_{\rm Ross}$ is the Rosseland mean opacity. The only change from the earlier wavelength-by-wavelength definition is the replacement

$$
\kappa_\lambda \longrightarrow \kappa_{\rm Ross}.
$$

**Why this particular mean?** Deep in an optically thick atmosphere, radiation escapes by diffusion. In that limit, the wavelengths that carry energy most efficiently are the relatively transparent windows, not the most opaque wavelengths. The Rosseland mean captures that by averaging the inverse opacity:

$$
\frac{1}{\kappa_{\rm Ross}}
=
\frac{\int_0^\infty \kappa_\lambda^{-1}\,(\partial B_\lambda/\partial T)\,d\lambda}
{\int_0^\infty(\partial B_\lambda/\partial T)\,d\lambda}.
$$

The factor $\partial B_\lambda/\partial T$ says which wavelengths respond most strongly to a local temperature change, and the factor $\kappa_\lambda^{-1}$ gives extra weight to wavelengths where photons travel farther before interacting. This is why the Rosseland mean is a harmonic, diffusion-weighted opacity, not a simple arithmetic average.

**Why it is convenient for spectra.** A spectrum calculation eventually computes a different optical depth $\tau_\lambda$ at each wavelength. The atmosphere itself, however, is stored as one table of physical conditions. $\tau_{\rm Ross}$ provides that table's depth coordinate: it orders layers by how deeply they sit in the diffusion problem, while still letting each wavelength form at its own $\tau_\lambda\sim1$ surface during spectral synthesis.

In the strictly grey atmosphere built here, $\kappa_\lambda$ is independent of wavelength, so $\kappa_{\rm Ross}$ reduces to that same grey opacity and $\tau_{\rm Ross}$ is just the single optical-depth coordinate $\tau$ used in the derivation above. We keep the Rosseland name because it is the same depth coordinate used once the opacity is allowed to vary with wavelength.

For the solar example, evaluate $T(\tau_{\rm Ross})$ on an 80-layer logarithmic grid. The grid runs from optically thin surface layers to optically thick diffusion layers. Logarithmic spacing is convenient because the photosphere occupies a narrow region around $\tau_{\rm Ross}\sim1$, while the useful atmosphere spans many orders of magnitude in optical depth.

![A first atmosphere is a depth grid plus physical conditions on that grid: temperature rises inward, and pressure rises because deeper layers support more overlying material.](resources/figures/s1_atmosphere_structure.png)""")

code(r'''def grey_temperature(teff_K, optical_depth):
    """Grey-atmosphere temperature T(tau) using the Kurucz/ATLAS Hopf fit."""
    effective_temperature = to_device(teff_K)
    tau_value = to_device(optical_depth)
    hopf_bracket = 0.710 + tau_value - 0.1331 * torch.exp(-3.4488 * tau_value)
    return effective_temperature * (0.75 * hopf_bracket)**0.25


teff_sun = to_device(5770.0)
logg_sun = to_device(4.44)
surface_gravity = 10.0 ** logg_sun

# 80 layers, equally spaced in log10(tau_Ross), from optically thin surface
# to deep optically thick layers.
layer_index = torch.arange(80, dtype=FLOAT_TYPE, device=EXECUTION_DEVICE)
rosseland_tau = 10.0 ** (-6.875 + 0.125 * layer_index)
temperature = grey_temperature(teff_sun, rosseland_tau)

photosphere_tau = to_device(2.0 / 3.0)
photosphere_temperature = grey_temperature(teff_sun, photosphere_tau)

print(f"layers: {temperature.numel()}   tau: {host_float(rosseland_tau[0]):.2e} .. {host_float(rosseland_tau[-1]):.2e}")
print(
    f"T(top) = {host_float(temperature[0]):.1f} K    "
    f"T(tau=2/3) = {host_float(photosphere_temperature):.1f} K    "
    f"T(bottom) = {host_float(temperature[-1]):.1f} K"
)''')


md(r"""The grid spans nearly ten decades in optical depth. The top layers are optically thin and cool. The deep layers are optically thick and hotter because radiation diffuses outward through overlying material. The photospheric value is close to $T_{\rm eff}$, matching the physical meaning of effective temperature.""")

code(r'''log_rosseland_tau = torch.log10(rosseland_tau)

plt.plot(
    display_array(log_rosseland_tau),
    display_array(temperature),
    color="C3",
)

log_photosphere_tau = torch.log10(photosphere_tau)
plt.axvline(host_float(log_photosphere_tau), ls="--", color="0.4", lw=1)
plt.text(host_float(log_photosphere_tau) + 0.1, host_float(torch.min(temperature)) + 300.0,
         r"photosphere, $\tau=2/3$", color="0.3")

plt.xlabel(r"$\log_{10}\tau_{\rm Ross}$")
plt.ylabel("temperature  [K]")
plt.title(r"Grey temperature structure of the Sun ($T_{\rm eff}=5770$ K)")
plt.tight_layout()
plt.show()''')


md(r"""Two immediate consequences should be visible in the numbers: temperature increases inward, and the photospheric temperature is close to $T_{\rm eff}$.""")

code(r'''temperature_step = temperature[1:] - temperature[:-1]
temperature_tolerance_K = to_device(1e-3)
temperature_increases_inward = bool(torch.all(temperature_step >= -temperature_tolerance_K).detach().cpu())
photosphere_temperature_ratio = photosphere_temperature / teff_sun

print(f"temperature increases inward: {temperature_increases_inward}")
print(f"T(tau=2/3) / T_eff = {host_float(photosphere_temperature_ratio):.4f}")''')


md(r"""## Hydrostatic Equilibrium: Pressure and Column Mass

Temperature alone is not an atmosphere. We also need pressure, because pressure sets density and controls many line-broadening and ionization effects.

Pressure is force per unit area. In a static atmosphere, the upward pressure force on a layer balances the downward weight of material above it. In a plane-parallel slab this balance is one-dimensional:

$$
\frac{dP_{\rm total}}{dz} = -\rho g ,
$$

where $P_{\rm total}$ is gas plus radiation pressure, $\rho$ is mass density, and $g$ is surface gravity. Column mass $m$ is the mass above one square centimetre of the stellar surface. In terms of $m$, hydrostatic equilibrium becomes especially simple:

$$
\frac{dP_{\rm total}}{dm} = g,
\qquad\Rightarrow\qquad
P_{\rm total}=gm
$$

if the pressure is measured relative to the top boundary.

**Hydrostatic equilibrium on an optical-depth grid.** The equation above naturally gives pressure as a function of column mass. The temperature law, however, is written as a function of Rosseland optical depth. Reusing the Rosseland relation defined above,

$$
d\tau_{\rm Ross}=\kappa_{\rm Ross}\,dm .
$$

Combining this with $dP_{\rm total}/dm=g$ gives

$$
\frac{dP_{\rm total}}{d\tau_{\rm Ross}}
=
\frac{g}{\kappa_{\rm Ross}} .
$$

This equation explains why opacity matters for pressure at a given optical depth. If $\kappa_{\rm Ross}$ is large, little mass is needed to reach $\tau_{\rm Ross}\sim1$, so the pressure there is smaller. If $\kappa_{\rm Ross}$ is small, more mass lies above the same optical depth, so the pressure is larger.

**Why use $\kappa_{\rm Ross}=1$ here?** This first grey atmosphere has not yet computed a physical opacity table. The grey temperature law supplies $T$ as a function of optical depth, but it does not by itself say how many grams per square centimetre correspond to that optical depth. To attach a first pressure scale, choose a constant unit opacity,

$$
\kappa_{\rm Ross}=1\ \mathrm{cm^2\,g^{-1}}.
$$

This is a normalization for the first model, not a claim that the Sun's Rosseland opacity is really $1\ \mathrm{cm^2\,g^{-1}}$. If a different constant opacity $\kappa_0$ were chosen, the same temperature law would give

$$
m=\frac{\tau_{\rm Ross}}{\kappa_0},
\qquad
P_{\rm total}=\frac{g\tau_{\rm Ross}}{\kappa_0}.
$$

With the unit choice $\kappa_0=1$, this reduces numerically to

$$
m=\tau_{\rm Ross},
\qquad
P_{\rm total}=g\tau_{\rm Ross}.
$$

The total pressure contains gas pressure plus radiation pressure. Radiation pressure is the momentum flux carried by an isotropic radiation field:

$$
P_{\rm rad}=\frac{4\sigma}{3c}T^4.
$$

In a solar photosphere this correction is small compared with gas pressure. Since the hydrostatic pressure above was measured relative to the top boundary, the radiation-pressure correction is also measured relative to its top value:

$$
\Delta P_{\rm rad}
=
\frac{4\sigma}{3c}\left[T^4(\tau)-T^4(\tau_{\rm top})\right],
\qquad
P_{\rm gas}=P_{\rm total}-\Delta P_{\rm rad}.
$$

A local mass density would require an equation of state. For this first atmosphere, pressure and column mass are enough to define the hydrostatic scale.""")

code(r'''# Unit-opacity grey atmosphere: kappa_Ross = 1, so column_mass = tau.
total_pressure = surface_gravity * rosseland_tau
column_mass = total_pressure / surface_gravity

# Small radiation-pressure correction, measured relative to the top boundary.
radiation_pressure = radiation_pressure_coefficient * (temperature**4 - temperature[0]**4)

gas_pressure = total_pressure - radiation_pressure

sample_layers = to_device([0, 20, 40, 60, 79], dtype=torch.long)
structure_sample = torch.stack((
    torch.log10(rosseland_tau.index_select(0, sample_layers)),
    temperature.index_select(0, sample_layers),
    gas_pressure.index_select(0, sample_layers),
    column_mass.index_select(0, sample_layers),
), dim=1)

structure_sample_np = display_array(structure_sample)
print("columns: log10(tau),  T [K],  P_gas [dyn/cm2],  column mass [g/cm2]")
print(np.array2string(structure_sample_np, precision=4))''')


md(r"""Reading the table downward is reading inward through the photosphere. The optical depth increases, the temperature rises, the gas pressure rises by many orders of magnitude, and the column mass increases because there is more material above each deeper layer.

The next two numbers simply report the normalization we chose: with $\kappa_{\rm Ross}=1$, column mass equals $\tau_{\rm Ross}$ and total pressure equals $g\tau_{\rm Ross}$ before the small radiation-pressure subtraction is applied to obtain $P_{\rm gas}$.""")

code(r'''max_column_mass_error = torch.max(torch.abs(column_mass / rosseland_tau - 1.0))
max_total_pressure_error = torch.max(torch.abs(total_pressure / (surface_gravity * rosseland_tau) - 1.0))

print(f"largest |column_mass/tau - 1| = {host_float(max_column_mass_error):.2e}")
print(f"largest |P_total/(g*tau) - 1| = {host_float(max_total_pressure_error):.2e}")''')


code(r'''fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))

log_tau_for_plot = display_array(torch.log10(rosseland_tau))
log_gas_pressure = display_array(torch.log10(gas_pressure))
log_column_mass = display_array(torch.log10(column_mass))

axes[0].plot(log_tau_for_plot, log_gas_pressure, color="C0")
axes[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$")
axes[0].set_ylabel(r"$\log_{10}\,P_{\rm gas}$  [dyn cm$^{-2}$]")
axes[0].set_title("Gas pressure")

axes[1].plot(log_tau_for_plot, log_column_mass, color="C2")
axes[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$")
axes[1].set_ylabel(r"$\log_{10}\,m$  [g cm$^{-2}$]")
axes[1].set_title(r"Column mass ($\kappa_{\rm Ross}=1$)")

fig.suptitle("Grey solar atmosphere: pressure and column mass")
fig.tight_layout()
plt.show()''')


md(r"""## What Has Been Built

Starting from $T_{\rm eff}=5770\,\mathrm{K}$ and $\log g=4.44$, we have built a self-contained first atmosphere. The Planck function supplies the thermal radiation scale. Optical depth supplies the natural depth coordinate for photon escape. LTE connects the local temperature to the thermal source function. The grey temperature relation supplies $T(\tau_{\rm Ross})$. Hydrostatic equilibrium supplies the first pressure and column-mass scale after we choose the unit-opacity normalization $\kappa_{\rm Ross}=1$.

The resulting atmosphere is a table of layers. Each layer has an optical depth, a temperature, a gas pressure, and a column mass. Reading downward in the table means moving inward through the photosphere: photons have farther to travel before escape, the material is hotter, and the pressure is larger because more mass lies above the layer.

This is not yet a detailed stellar spectrum. It has no wavelength-dependent opacity, no line opacity, no electron-density solution, and no radiative-transfer integral through a line-forming atmosphere. It does have the essential physical scaffold: a temperature structure on a Rosseland-depth grid and a first pressure scale tied to an explicit opacity normalization.""")


md(r"""## Practice Exercises

**1. Wien's law from the Planck function.** Compute $B_\lambda(T)$ on a fine wavelength grid for $T=3000,\ 5770,\ 10000\,\mathrm{K}$. Find the peak wavelength for each temperature and verify that $\lambda_{\rm peak}T\approx2.898\times10^6\,\mathrm{nm\,K}$.

**2. Photospheric temperature scale.** Evaluate the grey temperature relation at $\tau=1/2$, $2/3$, and $1$. How much does $T/T_{\rm eff}$ change over this range? What does that imply about the temperature contrast sampled by weak and strong spectral features?

**3. Surface gravity.** Recompute the pressure structure for the same $T_{\rm eff}$ but $\log g=4.0$ and $\log g=5.0$. At fixed optical depth, how does $P_{\rm gas}$ scale with $g$ in this simplified grey atmosphere?""")


md(r"""## Further Reading

- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed.** A clear observational and physical treatment of photospheres, line formation, and stellar spectra.
- **Mihalas, D. (1978). *Stellar Atmospheres*, 2nd ed.** A rigorous derivation of radiative transfer, LTE, grey atmospheres, and the Eddington approximation.
- **Hubeny, I. & Mihalas, D. (2014). *Theory of Stellar Atmospheres*.** A modern and comprehensive reference for stellar-atmosphere theory.""")


nb = new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
