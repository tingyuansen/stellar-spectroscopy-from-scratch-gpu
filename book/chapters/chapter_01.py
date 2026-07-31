"""Chapter 1: From starlight to a first grey atmosphere."""

from __future__ import annotations

from book.cells import code, markdown, notebook, setup_cell, source_code


def build_notebook() -> dict:
    """Construct Chapter 1 in the book's reference pedagogical style."""

    cells = [
        markdown(
            r"""
            # From Starlight to a First Grey Atmosphere

            *Stellar Spectroscopy from Scratch — from physical principles to a working code*

            Look at a high-resolution stellar spectrum. It may seem to be only a curve: brightness
            along the vertical direction, wavelength along the horizontal direction. But the curve
            contains two kinds of structure. A broad continuum changes smoothly with wavelength,
            while many narrow absorption features interrupt it. The continuum remembers a thermal
            scale. The lines remember which particles are present, how their internal states are
            populated, and what happened to the photons before they escaped.

            This leads to the question that organizes the book:

            > Given a physical description of a star, how do we predict the flux that leaves its
            > surface at every wavelength?

            We will answer it by building the calculation from the inside out. The first object is
            not a line list or a GPU kernel. It is a stack of atmospheric layers, with a
            temperature and pressure assigned to each depth. By the end of this chapter that stack
            will still be simple, but every number in it will have a physical reason for being
            there.

            **What this chapter will build**

            You will connect effective temperature to the local Planck function, derive optical
            depth from a mean free path, use LTE and radiative equilibrium to obtain a grey
            temperature profile, and use hydrostatic balance to attach a pressure scale. The result
            is four arrays—`standard_rosseland_optical_depth`, `temperature`, `column_mass`, and
            `gas_pressure`—each with shape `(80,)`.

            The arrays use 64-bit floating-point numbers on the CPU. The exact Planck demonstration
            uses a Torch array with axes `[depth, wavelength]`; the atmosphere scaffold uses NumPy
            arrays indexed from the outermost layer inward. Chapter 2 will explain those numerical
            choices. This chapter only asks that their meanings be explicit.

            > **Honest boundary.** We will not yet compute chemical populations, electron density,
            > physical opacity, a converged atmosphere, or a spectrum. No prior astronomy,
            > radiative transfer, or GPU programming is assumed.
            """
        ),
        setup_cell(),
        markdown(
            r"""
            ## 1.1 What are we trying to predict?

            An observation gives us flux as a function of wavelength. Working backward from that
            curve to the properties of the star is an **inverse problem**. Here we first solve the
            more controlled **forward problem**:

            $$
            \text{physical conditions}
            \quad\longrightarrow\quad
            \text{atmosphere}
            \quad\longrightarrow\quad
            F_\lambda(\lambda).
            $$

            \(F_\lambda\) is the energy crossing one square centimetre of the stellar surface per
            second and per wavelength interval, after averaging over the outward directions. Its
            broad continuum and narrow absorption lines are not two unrelated decorations. They
            are both records of how radiation was created and filtered at different depths.

            We know one global constraint before we know any of those depths. The **effective
            temperature**, \(T_{\rm eff}\), is defined by the total surface flux:

            $$
            F=\sigma T_{\rm eff}^4,
            $$

            where \(\sigma\) is the Stefan–Boltzmann constant. A star with
            \(T_{\rm eff}=5772\ {\rm K}\) releases the same total flux as a 5772 K blackbody.
            That does **not** mean every layer is at 5772 K. If all layers had one temperature,
            every wavelength would carry information from the same thermal state, and the spectrum
            would have far less structure than the one we observe.

            This is the chapter's central claim:

            > A stellar spectrum is not emitted by one mathematical surface. It is assembled from
            > radiation that samples a depth-dependent atmosphere.

            We will earn that statement by constructing the smallest atmosphere that has a depth
            coordinate, a temperature gradient, and a pressure scale.

            ## 1.2 Why the atmosphere comes before the spectrum

            To predict a spectrum, we must separate two tasks that answer different questions.

            A **model atmosphere** assigns temperature, pressure, density, particle populations,
            and related quantities to a sequence of depths. It answers questions such as “what is
            the electron density in layer 30?” and “how hot is the gas where the optical
            continuum escapes?”

            **Spectral synthesis** holds that atmosphere fixed, constructs the opacity at many
            wavelengths, and solves the transfer of radiation through the layers. It answers
            questions such as “how much flux emerges at 500.1 nm?”

            The atmosphere is therefore the stage on which a spectrum forms. Without it, saying
            that a strong line forms above a weak line has no quantitative meaning: there is no
            depth scale and no temperature or pressure attached to either location.
            """
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch01-forward-problem-v1.png"
                   alt="Original hand-sketched textbook diagram connecting a stellar photosphere to a one-dimensional atmosphere state, opacity and radiative transfer, and an emergent absorption spectrum.">
            <figcaption><strong>Conceptual schematic.</strong> A spectrum calculation replaces
              the star by a depth-dependent atmosphere state, then asks how opacity and radiative
              transfer turn that state into the flux that escapes. The drawing organizes the
              dependencies; its layer widths and spectral lines are not measured data.</figcaption>
            </figure>

            The arrows also expose the order in which we must learn the problem. Opacity and
            transfer consume an atmosphere; they cannot define one by themselves. We will
            therefore postpone the absorption lines in the final plot and first construct the
            layered state at the centre of the diagram.
            """
        ),
        markdown(
            r"""
            ## 1.3 Turn the photosphere into a stack of layers

            We model a small patch of a stellar photosphere as a vertical stack of layers.
            “Photosphere” means the region from which most of the observed light escapes; it is not
            a solid surface. The first task is to choose a geometry simple enough to calculate and
            rich enough to retain a depth gradient.

            Three geometric assumptions make that possible.

            **One dimensional.** Conditions vary with depth, but every point within one horizontal
            layer is treated as equivalent.

            **Plane parallel.** The layers are flat. This is a good approximation when the
            line-forming region is thin compared with the stellar radius.

            **Static.** The structure has no time dependence and no bulk acceleration. Microscopic
            thermal motion and microturbulent broadening remain; an expanding wind does not.

            These are controlled assumptions, not universal truths. Granulation needs three
            dimensions, pulsation needs time dependence, and an extended stellar wind needs a
            moving, often spherical atmosphere. We will introduce thermodynamic and mechanical
            assumptions only when the calculation reaches the equations that require them.
            """
        ),
        markdown(
            r"""
            Array index 0 will denote the outermost layer; increasing index moves inward. The
            production atmosphere uses 80 layers, but the number alone does not yet tell us where
            those layers belong. We first need a coordinate that measures escape, then a
            temperature law on that coordinate, and finally a pressure scale. Those three needs set
            the order of the next sections.
            """
        ),
        markdown(
            r"""
            ## 1.4 Units, constants, and one optical photon

            Temperature matters only because it sets an energy scale. Before writing a radiation
            law, compare the energy of one visible photon with the thermal energy available in a
            solar-temperature layer.

            Stellar-atmosphere formulae are traditionally written in **CGS units**: centimetres,
            grams, seconds, ergs, and dynes. The convention is practical. Opacity is commonly
            tabulated in cm\(^2\) g\(^{-1}\), pressure in dyn cm\(^{-2}\), and number density in
            cm\(^{-3}\). We will keep those units explicit rather than rely on memory.

            A photon of wavelength \(\lambda\) has energy \(E_\gamma=hc/\lambda\). Matter at
            temperature \(T\) has the thermal energy scale \(kT\). The exact constant names used by
            the implementation are:

            | implementation name | symbol | value |
            | --- | --- | ---: |
            | `PLANCK_ERG_SECOND` | \(h\) | \(6.62607015\times10^{-27}\) erg s |
            | `LIGHT_SPEED_CM_PER_S` | \(c\) | \(2.99792458\times10^{10}\) cm s\(^{-1}\) |
            | `BOLTZMANN_ERG_PER_K` | \(k\) | \(1.380649\times10^{-16}\) erg K\(^{-1}\) |

            The code asks a concrete question: how many units of \(kT\) does a 500 nm photon cost
            when `effective_temperature = 5772.0` K? It also prints energies in electron-volts:
            one electron-volt (eV) is the energy gained by one electron crossing a potential
            difference of one volt, equal to \(1.602176634\times10^{-12}\) erg.
            """
        ),
        code(
            """
            import numpy as np

            from payne_zero_synthesis.constants import (
                BOLTZMANN_ERG_PER_K,
                LIGHT_SPEED_CM_PER_S,
                PLANCK_ERG_SECOND,
            )

            effective_temperature = 5772.0
            wavelength_nm = 500.0
            wavelength_cm = wavelength_nm * 1.0e-7

            photon_energy_erg = (
                PLANCK_ERG_SECOND * LIGHT_SPEED_CM_PER_S / wavelength_cm
            )
            thermal_energy_erg = BOLTZMANN_ERG_PER_K * effective_temperature
            erg_per_electron_volt = 1.602176634e-12

            print(f"E_gamma(500 nm) = {photon_energy_erg / erg_per_electron_volt:.3f} eV")
            print(f"kT at 5772 K    = {thermal_energy_erg / erg_per_electron_volt:.3f} eV")
            print(f"E_gamma / kT    = {photon_energy_erg / thermal_energy_erg:.3f}")
            """
        ),
        markdown(
            r"""
            A 500 nm photon carries about \(2.5\) eV, while \(kT\) in a solar-temperature gas is
            about \(0.5\) eV. The ratio is close to five. Optical photons therefore probe an energy
            range in which thermal emission is sensitive to temperature and in which many atomic
            excitation and ionization energies matter.

            ## 1.5 Thermal radiation: the Planck function

            A layer does not emit only upward. Radiation has a direction as well as a wavelength.
            The quantity that retains this directional information is the **specific intensity**.
            \(I_\nu\) measures energy per area, time, frequency interval, and solid angle.
            A solid angle measures angular area on a unit sphere; its unit is the steradian, and a
            complete sphere contains \(4\pi\) steradians. This directional bookkeeping is why a
            factor \(4\pi\) will appear when an angular moment becomes a physical flux.

            Matter in thermodynamic equilibrium emits the blackbody intensity

            $$
            B_\nu(T)
            =
            \frac{2h\nu^3}{c^2}
            \frac{1}{\exp(h\nu/kT)-1}.
            $$

            Its CGS unit is erg s\(^{-1}\) cm\(^{-2}\) Hz\(^{-1}\) sr\(^{-1}\).
            The factor \(2\nu^2/c^2\) counts electromagnetic modes, \(h\nu\) supplies the energy
            per photon, and the denominator gives their thermal occupation. The dimensionless
            ratio

            $$
            x \equiv \frac{h\nu}{kT}
            $$

            is the photon energy measured in units of the local thermal energy. The corresponding
            implementation array is `photon_energy_over_thermal_energy`.

            Before evaluating the formula, its two limits tell us what a correct curve must do. If
            \(x\ll1\), then \(e^x-1\simeq x\), so

            $$
            B_\nu\simeq\frac{2\nu^2kT}{c^2}.
            $$

            This **Rayleigh–Jeans limit** grows linearly with temperature. If \(x\gg1\), then

            $$
            B_\nu\simeq\frac{2h\nu^3}{c^2}e^{-x},
            $$

            the **Wien limit**, where the intensity is exponentially sensitive to temperature.
            We should therefore expect hotter curves to brighten everywhere and their peaks to
            move toward shorter wavelengths. The plot will test those predictions.
            """
        ),
        markdown(
            r"""
            Spectra are often plotted per unit wavelength rather than per unit frequency. We
            cannot obtain \(B_\lambda\) by merely replacing \(\nu\) with \(c/\lambda\), because a
            one-hertz bin and a one-nanometre bin do not contain the same interval of radiation.
            Conservation of energy in a bin requires

            $$
            B_\lambda\,|d\lambda| = B_\nu\,|d\nu|.
            $$

            Since \(\nu=c/\lambda\),

            $$
            \left|\frac{d\nu}{d\lambda}\right|=\frac{c}{\lambda^2},
            \qquad
            B_\lambda = B_\nu\frac{c}{\lambda^2}.
            $$

            This Jacobian will reappear when synthesis converts its emergent flux per frequency to
            the public `flux_total` and `flux_continuum` per nanometre.

            A literal evaluation of \(\exp(x)-1\) is also numerically awkward: it can overflow when
            \(x\) is large. Multiplying numerator and denominator by \(\exp(-x)\) gives the
            identical expression

            $$
            \frac{1}{e^x-1}=\frac{e^{-x}}{1-e^{-x}},
            $$

            whose exponential is never larger than one. The next cell is the exact small
            `planck_bnu` function used by the synthesis kernel. Before reading it, note the
            contract:

            - `wavelength_nm` has shape `(wavelength,)`;
            - `temperature` has shape `(depth,)`;
            - the returned tensor has shape `(depth, wavelength)`;
            - `None` inserts the missing axis so Torch broadcasts every depth against every
              wavelength.

            A **tensor** is simply a multidimensional numerical array. Here all three tensors live
            on the CPU and use `torch.float64`, meaning 64-bit floating-point values. The axis
            order belongs to this function; another kernel may deliberately choose a different
            layout.

            This is the first point where our derivation meets an exact production kernel. Payne
            Zero uses the exact \(h\), \(c\), and \(k\) above to form
            `photon_energy_over_thermal_energy`, but retains the separately pinned amplitude
            `PLANCK_PREFACTOR = 1.47439e-2`. That literal is slightly rounded relative to
            recomputing \(2h(10^{15}\,{\rm Hz})^3/c^2\). Keeping it is a numerical compatibility
            choice, not new physics. The function below is short enough to read and execute in
            full.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import torch

            from book.plot_style import PAPER_COLORS, add_quiet_grid, single_panel
            from payne_zero_synthesis.constants import LIGHT_SPEED_NM_PER_S
            from payne_zero_synthesis.radiative_transfer import PLANCK_PREFACTOR
            """,
            tags=("book-setup", "hide-input"),
        ),
        source_code(
            "src/payne_zero_synthesis/radiative_transfer.py",
            ["planck_bnu"],
        ),
        markdown(
            r"""
            The function contains no hidden loop over wavelengths or depths. The two input axes
            are made explicit, and Torch evaluates the rectangular grid in one tensor operation.
            This `[depth, wavelength]` layout is exact for `planck_bnu`; it is not a universal
            synthesis convention. Chapter 2 will make each kernel's axis order explicit before
            arrays are combined.

            We can now examine the thermal spectrum. The cell converts the returned \(B_\nu\) to
            \(B_\lambda\) per nanometre using the Jacobian above. The calculation and the plot are
            kept in separate cells: first form and check the arrays, then decide how to display
            them.
            """
        ),
        code(
            """
            wavelength_nm = torch.linspace(200.0, 2000.0, 700, dtype=torch.float64)
            temperature = torch.tensor([4500.0, 5772.0, 7500.0], dtype=torch.float64)

            planck_per_frequency = planck_bnu(wavelength_nm, temperature)
            planck_per_wavelength_nm = (
                planck_per_frequency
                * LIGHT_SPEED_NM_PER_S
                / wavelength_nm[None, :] ** 2
            )

            assert planck_per_wavelength_nm.shape == (3, 700)
            assert torch.all(torch.diff(planck_per_wavelength_nm, dim=0) > 0.0)
            """
        ),
        markdown(
            r"""
            The array now has three temperature rows and 700 wavelength columns. The figure uses
            three curves in one panel because they all answer the same question: how does thermal
            emission change when the gas becomes hotter?
            """
        ),
        code(
            """
            figure, axes = single_panel()
            colors = (
                PAPER_COLORS["blue"],
                PAPER_COLORS["orange"],
                PAPER_COLORS["green"],
            )
            for index, color in enumerate(colors):
                axes.plot(
                    wavelength_nm.numpy(),
                    planck_per_wavelength_nm[index].numpy(),
                    color=color,
                    label=f"{temperature[index]:.0f} K",
                )
            axes.set(
                xlabel="wavelength  [nm]",
                ylabel=r"$B_\\lambda$  [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ nm$^{-1}$]",
                title="Hotter thermal spectra brighten and shift blueward",
                xlim=(200.0, 2000.0),
                ylim=(0.0, None),
            )
            axes.legend(title="temperature")
            add_quiet_grid(axes, axis="y")
            plt.show()
            plt.close(figure)
            """
        ),
        markdown(
            r"""
            The plot confirms both predictions. At any fixed optical wavelength, hotter material
            emits more strongly, and the peak moves blueward as temperature rises. The array check
            below the plot also verifies the expected shape; the smooth positive curves rule out a
            sign error in the exponential.

            We can now compute what radiation a layer can supply locally. We still cannot say
            whether that radiation reaches space. A photon created deep in the atmosphere may be
            absorbed or scattered many times before it can escape, which creates the need for a
            depth coordinate based on interaction rather than kilometres.
            """
        ),
        markdown(
            r"""
            ## 1.6 Optical depth measures the difficulty of escape

            The **mass extinction coefficient** \(\kappa_\lambda\) is the effective area that
            removes radiation from a beam per gram of material. Its unit is cm\(^2\) g\(^{-1}\).
            Extinction includes true absorption and **scattering**, in which a photon is redirected
            rather than simply destroyed. Multiplying by mass density \(\rho\), measured in
            g cm\(^{-3}\), gives
            \(\kappa_\lambda\rho\) in cm\(^{-1}\): an interaction probability per unit path
            length.

            The inverse is the photon mean free path,

            $$
            \ell_\lambda=\frac{1}{\kappa_\lambda\rho}.
            $$

            A photon does not primarily care how many geometric kilometres remain. It cares how
            many mean free paths remain. That observation motivates **optical depth**.

            Let geometric height \(z\) increase outward. We define

            $$
            d\tau_\lambda=-\kappa_\lambda\rho\,dz.
            $$

            The minus sign makes \(\tau_\lambda\) increase inward. A deeper starting point has more
            material between it and space.
            """
        ),
        markdown(
            r"""
            Consider a beam that only loses photons, with no local emission added. Over a small
            optical-depth interval, its fractional loss is

            $$
            dI_\lambda=-I_\lambda\,d\tau_\lambda.
            $$

            Dividing by \(I_\lambda\) and integrating from the surface to optical depth \(\tau\)
            gives

            $$
            \int_{I_0}^{I}\frac{dI_\lambda}{I_\lambda}
            =
            -\int_0^\tau d\tau_\lambda,
            \qquad
            I=I_0e^{-\tau}.
            $$

            Thus optical depth has an immediate operational meaning. At \(\tau=1\), only
            \(e^{-1}\simeq0.37\) of an un-replenished vertical beam survives. At
            \(\tau\ll1\) the layer is transparent; at \(\tau\gg1\) direct escape is strongly
            suppressed.

            The following one-panel figure shows this transition. The dashed guide marks
            \(\tau=1\); it is not a fitted boundary or a solid surface.
            """
        ),
        code(
            """
            optical_depth = np.logspace(-3.0, 1.2, 500)
            transmitted_fraction = np.exp(-optical_depth)

            figure, axes = single_panel()
            axes.semilogx(
                optical_depth,
                transmitted_fraction,
                color=PAPER_COLORS["blue"],
            )
            axes.axvline(
                1.0,
                color=PAPER_COLORS["grey"],
                linestyle="--",
                linewidth=1.2,
            )
            axes.annotate(
                r"$\\tau=1$",
                xy=(1.0, np.exp(-1.0)),
                xytext=(1.45, 0.55),
                arrowprops={"arrowstyle": "-", "color": PAPER_COLORS["grey"]},
            )
            axes.set(
                xlabel=r"optical depth  $\\tau$",
                ylabel=r"transmitted fraction  $I/I_0$",
                title="Direct transmission falls exponentially with optical depth",
                ylim=(-0.02, 1.02),
            )
            add_quiet_grid(axes, axis="y")
            plt.show()
            plt.close(figure)
            """
        ),
        markdown(
            r"""
            The visible photosphere is therefore a region, not a material boundary. Radiation
            emerging along a ray with direction cosine
            \(\mu=\cos\theta\), where \(\theta\) is measured from the outward surface normal,
            travels a slanted path; its characteristic vertical depth is of order
            \(\tau_\lambda\sim\mu\). Flux averages over many outward directions, so “the
            photosphere is near optical depth one” is the robust statement.

            It is often more useful to describe depth by **column mass**,

            $$
            m(z)=\int_z^\infty\rho(z')\,dz',
            $$

            the mass above one square centimetre of surface. Because \(dm=-\rho\,dz\),

            $$
            d\tau_\lambda=\kappa_\lambda\,dm.
            $$

            This equation joins radiation to mechanics. Optical depth measures the path to escape;
            column mass measures the weight above a layer. It also shows why different wavelengths
            escape from different depths: \(\kappa_\lambda\) changes with wavelength. A strong
            line reaches \(\tau_\lambda\sim1\) with less overlying mass than a transparent
            continuum window.
            """
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch01-formation-depth-v1.png"
                   alt="Original hand-sketched textbook diagram of a plane-parallel atmosphere in which continuum, weak-line, and strong-line radiation sample progressively higher characteristic depths.">
              <figcaption><strong>Conceptual schematic.</strong> The continuum, a weak line, and a
              strong line reach optical depth of order unity at different characteristic depths.
              A strong line usually becomes opaque with less overlying mass than the nearby
              continuum. The wavy paths summarize the depth ranges sampled; they are not literal
              photon trajectories or infinitely thin formation layers.</figcaption>
            </figure>

            The drawing can now be read rather than taken on faith. Wavelength dependence enters
            through \(\kappa_\lambda\); column mass supplies the amount of material; their product
            determines which layers can communicate with the surface. We know how radiation is
            attenuated, but attenuation alone cannot explain a luminous star. The next missing
            piece is local emission along the path.
            """
        ),
        markdown(
            r"""
            ## 1.7 LTE supplies a local source, not a global blackbody

            Attenuation alone can only make a beam dimmer. A stellar atmosphere also emits. The
            **source function** \(S_\lambda\) is local emissivity divided by local extinction, so
            it has the same units as specific intensity. Along an outward ray, the
            plane-parallel transfer equation is

            $$
            \mu\frac{dI_\lambda}{d\tau_\lambda}
            =
            I_\lambda-S_\lambda.
            $$

            The two terms express a competition. Extinction tries to remove the current beam;
            local emission tries to replace it with the source function.

            LTE says that collisions keep the local material populations thermal at the local
            temperature \(T\). Kirchhoff's law then gives the thermal source for true absorption:

            $$
            S_\lambda=B_\lambda(T).
            $$

            This does **not** make the escaping spectrum a single blackbody. Temperature changes
            with depth, photons connect different layers, and scattering redirects radiation
            without necessarily thermalizing it. The implementation therefore keeps continuous
            absorption and continuous scattering separate.

            For a vertical ray through a slab with constant source \(S\), the transfer equation
            has the exact solution

            $$
            I_{\rm out}=I_{\rm in}e^{-\tau}+S(1-e^{-\tau}).
            $$

            The thin and thick limits make the role of the source function concrete.
            """
        ),
        code(
            """
            slab_optical_depth = np.logspace(-3.0, 2.0, 500)
            incident_intensity = 0.20
            source_function = 1.00
            outgoing_intensity = (
                incident_intensity * np.exp(-slab_optical_depth)
                + source_function * (1.0 - np.exp(-slab_optical_depth))
            )

            figure, axes = single_panel()
            axes.semilogx(
                slab_optical_depth,
                outgoing_intensity,
                color=PAPER_COLORS["orange"],
                label=r"$I_{\\rm out}$",
            )
            axes.axhline(
                source_function,
                color=PAPER_COLORS["black"],
                linestyle="--",
                linewidth=1.2,
                label=r"source $S$",
            )
            axes.set(
                xlabel=r"slab optical depth  $\\tau$",
                ylabel="intensity  [arbitrary units]",
                title="An optically thick LTE slab approaches its local source",
                ylim=(0.15, 1.05),
            )
            axes.legend()
            add_quiet_grid(axes, axis="y")
            plt.show()
            plt.close(figure)
            """
        ),
        markdown(
            r"""
            When the slab is thin, the incident beam is barely changed. When it is thick, the
            factor multiplying \(I_{\rm in}\) vanishes and the outgoing intensity approaches the
            local source. This is why photons that escape from different depths carry information
            about the temperature at those depths.

            ## 1.8 Radiative equilibrium gives a grey temperature law

            We now need a temperature at every depth. The simplest useful structure is a
            **grey atmosphere**, in which opacity is treated as independent of wavelength. A grey
            atmosphere cannot produce absorption lines. Its purpose is to reveal the thermal
            scaffold on which wavelength-dependent physics will later act.

            The atmosphere is in **radiative equilibrium**: after integrating over all wavelengths
            and directions, each layer passes along the same flux. No layer gains or loses net
            energy. The constant flux is fixed by the effective temperature:

            $$
            F=\sigma T_{\rm eff}^{4}.
            $$

            > **The one dense derivation in this chapter.** Keep hold of the causal chain:
            > constant flux fixes a slope, an angular closure connects the needed moments, and a
            > surface boundary fixes the intercept. The algebra below makes those three statements
            > precise.

            Solving the full intensity for every direction would be possible but unnecessary for
            this first structure. Instead, we compress the frequency-integrated—**bolometric**—
            intensity \(I(\mu)\) into a few angular summaries.

            First imagine only two vertical rays of equal intensity: one upward
            (\(\mu=+1\)) and one downward (\(\mu=-1\)). Their average intensity is nonzero, but
            their energy flows cancel. If the upward ray becomes brighter, the average changes only
            modestly while a net outward flux appears. We therefore need one average that ignores
            direction and another that retains its sign. A third weighting by \(\mu^2\) measures
            how strongly radiation carries vertical momentum.

            The continuous versions of those three summaries are

            $$
            J=\frac12\int_{-1}^{1} I(\mu)\,d\mu,
            $$

            $$
            H=\frac12\int_{-1}^{1} \mu I(\mu)\,d\mu,
            $$

            $$
            K=\frac12\int_{-1}^{1} \mu^2 I(\mu)\,d\mu.
            $$

            \(J\) is the mean-intensity moment, \(H\) is the flux moment, and \(K\) is the
            radiation-pressure moment. The two-ray example explains the weights: \(J\) counts
            radiation present, \(H\) counts the outward-minus-inward flow, and \(K\) retains the
            directionality of momentum. With these definitions, the physical flux is \(F=4\pi H\).
            """
        ),
        markdown(
            r"""
            Radiative equilibrium and grey LTE connect the local mean intensity to the integrated
            Planck function:

            $$
            B(T)
            \equiv
            \int_0^\infty B_\nu(T)\,d\nu
            =
            \frac{\sigma}{\pi}T^4,
            \qquad
            J=B(T).
            $$

            Here \(B(T)\) is the Planck function integrated over all frequencies, not a new
            monochromatic law. The equality \(J=B\) states local energy balance: the layer absorbs
            as much radiation as it thermally emits.

            This connection can also be seen directly from the transfer equation. Integrating that
            equation over angle gives the zeroth-moment equation

            $$
            \frac{dH}{d\tau}=J-S.
            $$

            In grey LTE, \(S=B\). Radiative equilibrium requires no net local heating, so
            \(J=S=B\), and therefore \(dH/d\tau=0\): the flux moment is constant with depth.
            Its value is

            $$
            H=\frac{F}{4\pi}
            =\frac{\sigma T_{\rm eff}^{4}}{4\pi}.
            $$

            Multiplying the transfer equation by \(\mu\) before integrating gives the next moment:

            $$
            \frac{dK}{d\tau}=H.
            $$

            We now know \(H\), but this equation contains \(K\) while the thermal relation is
            written in terms of \(J\). One more relationship is required. Such a relationship is
            called a **closure**. The Eddington approximation uses

            $$
            K\simeq\frac{J}{3}.
            $$

            The factor \(1/3\) is the angular average of \(\mu^2\) for an isotropic field. The
            approximation is best deep in the atmosphere and imperfect near the surface, where
            radiation streams preferentially outward. It is nevertheless the cleanest controlled
            closure for a first model.

            Substituting \(K=J/3\) into \(dK/d\tau=H\) gives

            $$
            \frac{1}{3}\frac{dJ}{d\tau}=H.
            $$

            Substituting the expressions for \(J\) and \(H\) gives

            $$
            \frac{dT^4}{d\tau}
            =\frac{3}{4}T_{\rm eff}^{4}.
            $$

            Radiative equilibrium therefore fixes the *slope* of \(T^4\) with optical depth.
            It does not fix the intercept. Integrating the moment equation gives

            $$
            J=3H(\tau+q),
            $$

            where \(q\) records the surface boundary. Space sends no radiation downward onto the
            atmosphere. In the Eddington approximation, that no-incoming-radiation boundary is
            represented by \(J(0)=2H\). Substitution gives \(3Hq=2H\), hence
            \(q=2/3\). The boundary condition, rather than radiative equilibrium itself, supplies
            the constant:

            $$
            T^4(\tau)
            =
            \frac{3}{4}T_{\rm eff}^{4}
            \left(\tau+\frac{2}{3}\right).
            $$

            At \(\tau=2/3\), this relation gives \(T=T_{\rm eff}\). That familiar depth is a
            consequence of the grey closure and its boundary condition, not a physical wall.
            """
        ),
        markdown(
            r"""
            ## 1.9 Rosseland depth puts every layer on one grid

            Real opacity varies enormously with wavelength, so the full calculation has a
            different \(\tau_\lambda\) at each wavelength. The atmosphere itself still needs one
            shared depth coordinate on which to store temperature and pressure.

            In optically thick layers, radiation moves by diffusion. Energy preferentially leaks
            through relatively transparent wavelength windows. A two-window example makes the
            needed average clear. Suppose two equally important wavelength bins have opacities
            \(1\) and \(100\ {\rm cm^2\,g^{-1}}\). Their arithmetic mean is \(50.5\), which
            describes neither the easy route nor the resulting leakage. Their harmonic mean is

            $$
            \left[\frac12\left(\frac11+\frac1{100}\right)\right]^{-1}
            \simeq1.98\ {\rm cm^2\,g^{-1}}.
            $$

            The transparent window dominates because that is where energy escapes most readily.
            A real spectrum replaces the two equal weights by the temperature response of the
            radiation field. The appropriate representative opacity is the weighted harmonic mean

            $$
            \frac{1}{\kappa_{\rm Ross}}
            =
            \frac{
            \int_0^\infty
            \kappa_\lambda^{-1}
            \left(\partial B_\lambda/\partial T\right)d\lambda
            }{
            \int_0^\infty
            \left(\partial B_\lambda/\partial T\right)d\lambda
            }.
            $$

            The inverse opacity gives transparent windows extra influence.
            \(\partial B_\lambda/\partial T\) emphasizes wavelengths that carry a strong response
            to a temperature change. The common coordinate is then defined by

            $$
            d\tau_{\rm Ross}=\kappa_{\rm Ross}\,dm.
            $$

            The stored standard coordinate is `standard_rosseland_optical_depth`. The next
            function is the exact grid constructor used by the atmosphere run setup. It produces
            `layers` values in `float64`, ordered from the outermost layer to the innermost layer.
            """
        ),
        source_code(
            "src/payne_zero_atmosphere/run_setup.py",
            ["standard_rosseland_optical_depth_grid"],
        ),
        markdown(
            r"""
            The standard atmosphere has 80 layers. The grid is uniform in
            \(\log_{10}\tau_{\rm Ross}\):

            $$
            \tau_{{\rm Ross},i}
            =
            10^{-6.875+0.125i},
            \qquad i=0,\ldots,79.
            $$

            We can now evaluate the Eddington grey law on this exact coordinate:

            $$
            T_i
            =
            T_{\rm eff}
            \left[
            \frac34\left(
            \tau_{{\rm Ross},i}+\frac23
            \right)
            \right]^{1/4}.
            $$
            """
        ),
        code(
            """
            standard_rosseland_optical_depth = (
                standard_rosseland_optical_depth_grid(80)
            )
            grey_temperature = effective_temperature * (
                0.75 * (standard_rosseland_optical_depth + 2.0 / 3.0)
            ) ** 0.25

            print(f"layers = {grey_temperature.size}")
            print(
                "Rosseland depth = "
                f"{standard_rosseland_optical_depth[0]:.3e} ... "
                f"{standard_rosseland_optical_depth[-1]:.3e}"
            )
            print(
                f"temperature = {grey_temperature[0]:.1f} ... "
                f"{grey_temperature[-1]:.1f} K"
            )
            """
        ),
        markdown(
            r"""
            The grid spans almost ten orders of magnitude. Its outer layers are optically thin;
            its inner layers lie in the diffusion regime. Logarithmic spacing resolves the
            transition near \(\tau_{\rm Ross}\sim1\) without wasting most layers at large optical
            depth.

            The plot below asks one question only: how does the grey temperature vary along this
            coordinate? The annotation at \(2/3\) connects the numerical profile to the boundary
            result derived above.
            """
        ),
        code(
            """
            figure, axes = single_panel()
            axes.plot(
                np.log10(standard_rosseland_optical_depth),
                grey_temperature,
                color=PAPER_COLORS["orange"],
            )
            photospheric_log_depth = np.log10(2.0 / 3.0)
            axes.vlines(
                photospheric_log_depth,
                4300.0,
                31500.0,
                color=PAPER_COLORS["grey"],
                linestyle="--",
                linewidth=1.2,
            )
            axes.annotate(
                r"$\\tau_{\\rm Ross}=2/3$",
                xy=(photospheric_log_depth, effective_temperature),
                xytext=(-2.3, 7200.0),
                arrowprops={"arrowstyle": "-", "color": PAPER_COLORS["grey"]},
            )
            axes.set(
                xlabel=r"$\\log_{10}\\tau_{\\rm Ross}$",
                ylabel="temperature  [K]",
                title="The Eddington grey temperature rises inward",
                ylim=(4300.0, 31500.0),
            )
            add_quiet_grid(axes, axis="y")
            plt.show()
            plt.close(figure)
            """
        ),
        markdown(
            r"""
            Temperature rises inward because deeper layers must carry the same outward flux through
            more overlying material. Evaluating the formula directly at
            \(\tau_{\rm Ross}=2/3\) should recover `effective_temperature`. This is an analytic
            identity, so it deserves a strict numerical check.
            """
        ),
        code(
            """
            photospheric_temperature = effective_temperature * (
                0.75 * (2.0 / 3.0 + 2.0 / 3.0)
            ) ** 0.25

            np.testing.assert_allclose(
                photospheric_temperature,
                effective_temperature,
                rtol=0.0,
                atol=1.0e-12,
            )
            assert np.all(np.diff(grey_temperature) > 0.0)
            print(f"T(tau_Ross=2/3) = {photospheric_temperature:.1f} K")
            print("temperature increases inward: yes")
            """
        ),
        markdown(
            r"""
            The calculation returns exactly \(5772.0\ {\rm K}\) at
            \(\tau_{\rm Ross}=2/3\), and the monotonicity check confirms that all 80 temperatures
            rise inward as predicted. We have therefore attached a temperature to every layer.
            Temperature alone cannot support the material against gravity, so the next step is to
            give those layers a pressure scale.
            """
        ),
        markdown(
            r"""
            ## 1.10 Hydrostatic balance supplies a pressure scale

            Temperature alone is not an atmosphere. Pressure controls density, ionization, and
            pressure broadening. The downward acceleration is the surface gravity \(g\). Its exact
            public coordinate is `log_surface_gravity = log10(g)`, with \(g\) in
            cm s\(^{-2}\). For a solar-like example we adopt 4.44 and first recover the physical
            acceleration.
            """
        ),
        code(
            """
            log_surface_gravity = 4.44
            surface_gravity_cgs = 10.0**log_surface_gravity

            print(f"surface gravity = {surface_gravity_cgs:.3e} cm s^-2")
            print(f"                = {surface_gravity_cgs / 100.0:.1f} m s^-2")
            """
        ),
        markdown(
            r"""
            The result is about \(275\ {\rm m\,s^{-2}}\), roughly 28 times Earth's surface
            gravity. Acceleration alone still does not give pressure; we also need the mass it
            supports. Column mass now supplies exactly that missing quantity.

            We introduce the final assumption of this first model: **hydrostatic balance**. The
            atmosphere is static, so the upward pressure force balances gravity:

            $$
            \frac{dP_{\rm total}}{dz}=-\rho g.
            $$

            Since \(dm=-\rho\,dz\), column mass turns this into

            $$
            \frac{dP_{\rm total}}{dm}=g.
            $$

            If pressure is measured relative to the top boundary and \(g\) is constant through the
            thin atmosphere,

            $$
            P_{\rm total}=gm.
            $$

            Combining hydrostatic balance with
            \(d\tau_{\rm Ross}=\kappa_{\rm Ross}dm\) gives

            $$
            \frac{dP_{\rm total}}{d\tau_{\rm Ross}}
            =\frac{g}{\kappa_{\rm Ross}}.
            $$

            This explains why pressure at a fixed optical depth depends on opacity. If
            \(\kappa_{\rm Ross}\) is large, little mass is needed to reach that depth. If it is
            small, more mass—and therefore more weight—lies above the layer.
            """
        ),
        markdown(
            r"""
            We have not yet computed a physical `rosseland_opacity`; that requires the equation of
            state and many absorption and scattering processes. To attach a transparent pressure
            scale to the grey temperature, we now declare the controlled limit

            $$
            \kappa_{\rm Ross}=1\ {\rm cm^2\,g^{-1}}.
            $$

            This is a unit normalization, not a solar-opacity estimate. In this limit,

            $$
            m=\frac{\tau_{\rm Ross}}{\kappa_{\rm Ross}}
            =\tau_{\rm Ross}\ {\rm g\,cm^{-2}},
            \qquad
            P_{\rm total}=gm.
            $$

            Radiation also carries momentum. This is the pressure-like information that the
            \(K\) moment retained in the grey derivation. For an isotropic thermal field,
            \(K=J/3\), and the corresponding pressure is

            $$
            P_{\rm rad}=\frac{4\sigma}{3c}T^4.
            $$

            Because our hydrostatic pressure is measured from the top boundary, we subtract only
            the *increase* in radiation pressure below the top:

            $$
            P_{\rm gas}
            =
            P_{\rm total}
            -
            \frac{4\sigma}{3c}
            \left(T^4-T_{\rm top}^4\right).
            $$

            The arrays in the next cell use the exact production field names wherever those fields
            already exist. They remain local arrays because this controlled limit has not computed
            enough physics to claim a complete `ModelAtmosphere`.
            """
        ),
        code(
            """
            rosseland_opacity = np.ones(80, dtype=np.float64)
            column_mass = standard_rosseland_optical_depth / rosseland_opacity
            temperature = grey_temperature.copy()

            total_pressure = surface_gravity_cgs * column_mass
            stefan_boltzmann_cgs = (
                2.0
                * np.pi**5
                * BOLTZMANN_ERG_PER_K**4
                / (15.0 * PLANCK_ERG_SECOND**3 * LIGHT_SPEED_CM_PER_S**2)
            )
            radiation_pressure_increment = (
                4.0
                * stefan_boltzmann_cgs
                / (3.0 * LIGHT_SPEED_CM_PER_S)
                * (temperature**4 - temperature[0] ** 4)
            )
            gas_pressure = total_pressure - radiation_pressure_increment

            assert np.all(gas_pressure > 0.0)
            assert np.all(np.diff(gas_pressure) > 0.0)
            """
        ),
        markdown(
            r"""
            A few widely separated layers are more informative than printing all 80 rows. Reading
            down the table means moving inward. `column_mass` and `gas_pressure` should both grow
            because progressively more material lies above the layer.
            """
        ),
        code(
            """
            sample_layer = np.array([0, 20, 54, 70, 79])
            structure_sample = np.column_stack(
                (
                    sample_layer,
                    np.log10(standard_rosseland_optical_depth[sample_layer]),
                    temperature[sample_layer],
                    gas_pressure[sample_layer],
                    column_mass[sample_layer],
                )
            )

            print("layer  log10(tau_Ross)   temperature [K]   "
                  "gas pressure [dyn cm^-2]   column mass [g cm^-2]")
            for row in structure_sample:
                print(
                    f"{int(row[0]):5d} {row[1]:16.3f} {row[2]:17.1f} "
                    f"{row[3]:25.4e} {row[4]:22.4e}"
                )
            """
        ),
        markdown(
            r"""
            The outer row contains only \(1.33\times10^{-7}\ {\rm g\,cm^{-2}}\) above it and needs
            just \(3.67\times10^{-3}\ {\rm dyn\,cm^{-2}}\) of gas pressure. Near the photosphere,
            layer 54 has \(\tau_{\rm Ross}\simeq0.75\) and already supports about
            \(2.07\times10^4\ {\rm dyn\,cm^{-2}}\). At the bottom, the overlying column has reached
            \(10^3\ {\rm g\,cm^{-2}}\), and the pressure is \(2.75\times10^7\ {\rm dyn\,cm^{-2}}\).
            That many-order-of-magnitude range calls for a logarithmic vertical axis in the
            one-panel view below.
            """
        ),
        code(
            """
            figure, axes = single_panel()
            axes.plot(
                np.log10(standard_rosseland_optical_depth),
                gas_pressure,
                color=PAPER_COLORS["blue"],
            )
            axes.set_yscale("log")
            axes.set(
                xlabel=r"$\\log_{10}\\tau_{\\rm Ross}$",
                ylabel=r"gas pressure  [dyn cm$^{-2}$]",
                title="Hydrostatic gas pressure rises inward",
            )
            add_quiet_grid(axes, axis="y")
            plt.show()
            plt.close(figure)
            """
        ),
        markdown(
            r"""
            The monotonic rise is the expected hydrostatic behavior. Two algebraic checks also
            expose the normalization: with unit `rosseland_opacity`, `column_mass` must equal
            `standard_rosseland_optical_depth`, and before the radiation-pressure subtraction,
            `total_pressure` must equal `surface_gravity_cgs * column_mass`.
            """
        ),
        code(
            """
            np.testing.assert_array_equal(
                column_mass,
                standard_rosseland_optical_depth,
            )
            np.testing.assert_allclose(
                total_pressure,
                surface_gravity_cgs * column_mass,
                rtol=0.0,
                atol=0.0,
            )

            pressure_fraction = radiation_pressure_increment[-1] / total_pressure[-1]
            print(f"bottom radiation-pressure correction = {pressure_fraction:.3e} of total")
            print("unit-opacity hydrostatic identities passed")
            """
        ),
        markdown(
            r"""
            The printed fraction is about \(7.6\times10^{-5}\). In this solar-like, unit-opacity
            scaffold, radiation pressure is therefore a small correction to the weight-supported
            pressure at the bottom layer. The subtraction is still physically required: the
            fraction grows rapidly with temperature through \(T^4\), and it also becomes more
            important when gravity supplies less total pressure. The two identity checks establish
            the algebra; the small fraction gives that algebra a physical scale.
            """
        ),
        markdown(
            r"""
            ## 1.11 Read the scaffold as a causal chain

            The calculation has not merely produced four arrays. It has connected them in a
            specific order:

            $$
            T_{\rm eff}
            \longrightarrow \tau_{\rm Ross}
            \longrightarrow T(\tau_{\rm Ross})
            \longrightarrow m(\tau_{\rm Ross})
            \longrightarrow P_{\rm gas}(\tau_{\rm Ross}).
            $$

            `effective_temperature` fixes the total outward energy flow. Optical depth supplies a
            coordinate tied to photon escape. Grey radiative equilibrium turns that coordinate
            into `temperature`. The declared unit opacity turns optical depth into `column_mass`,
            and gravity turns the overlying mass into pressure. Finally, subtracting the increase
            in radiation pressure leaves `gas_pressure`.

            This chain is useful because every assumption has a visible consequence:

            | assumption | what it allowed us to calculate | what it leaves unresolved |
            | --- | --- | --- |
            | grey opacity | one analytic temperature law | wavelength-dependent absorption |
            | LTE | a local thermal source scale | non-local excitation and ionization |
            | unit mean opacity | a transparent mass scale | opacity from the actual mixture |
            | hydrostatic balance | pressure from overlying weight | flows and accelerations |
            | plane-parallel, static layers | one ordered depth coordinate | geometry and time dependence |

            The four arrays are therefore a **scaffold**, not a finished atmosphere. We have not
            counted atoms, ions, molecules, or free electrons. We have not calculated which
            processes absorb or scatter light, nor allowed those opacities to change the
            temperature and pressure that produced them. Convection and radiative acceleration
            are also absent.

            Most importantly, no spectrum has been synthesized. A spectrum requires
            wavelength-dependent opacity and a transfer calculation through the layers. The
            scaffold nevertheless earns its place: it makes the dependencies visible before the
            later physics couples them into a much larger calculation.
            """
        ),
        markdown(
            r"""
            ## Further reading

            These references deepen the same physics; none is required to run or understand the
            construction above.

            - Gray, D. F. (2005), *The Observation and Analysis of Stellar Photospheres*, 3rd ed.,
              gives a physical path from photospheres to observed line profiles.
            - Mihalas, D. (1978), *Stellar Atmospheres*, 2nd ed., develops radiative transfer, LTE,
              angular moments, and grey atmospheres rigorously.
            - Hubeny, I. & Mihalas, D. (2014), *Theory of Stellar Atmospheres*, provides a modern
              and comprehensive treatment of atmosphere physics.
            """
        ),
        markdown(
            r"""
            ## 1.12 Chapter summary

            We began with the flux leaving a star and built the first atmosphere scaffold needed
            to predict it. `effective_temperature` fixed the total energy flow, the Planck function
            supplied the local thermal radiation scale, and optical depth measured how strongly
            matter obstructs escape. LTE connected local temperature to thermal emission without
            making the whole radiation field a blackbody. Grey radiative equilibrium then supplied
            a first `temperature` profile, while hydrostatic balance connected `column_mass`,
            gravity, and `gas_pressure`.

            The result is deliberately incomplete but honest: four self-contained arrays on an
            80-layer coordinate. We can now state precisely what is missing—electron density,
            chemical populations, physical opacity, radiation forces, convection, and the
            feedback that makes a stellar atmosphere self-consistent.

            ### Next: make the calculation trustworthy and fast

            The opening question still has no \(F_\lambda\). To obtain one, later chapters must
            integrate radiation through depth at many wavelengths while preserving units, axis
            meanings, and numerical order. A small mistake there could produce a smooth-looking
            spectrum from the wrong integral.

            Chapter 2 therefore begins with one concrete task: make the depth integration agree
            across a readable scalar calculation, NumPy, Numba, parallel CPU work, and Torch
            devices. That task gives us the array contracts, data provenance, precision rules, and
            reproducible checks needed before chemistry and opacity make the calculation much
            larger.

            [Continue to Chapter 2: From Equations to Fast, Trustworthy Kernels and Explicit
            Data](/reader.html?ch=2)
            """
        ),
    ]
    return notebook(cells)
