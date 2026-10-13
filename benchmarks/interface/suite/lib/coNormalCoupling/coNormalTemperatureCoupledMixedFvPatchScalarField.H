/*---------------------------------------------------------------------------*\
  compressible::coNormalTemperatureCoupledMixed  (OpenFOAM v2606, ESI)

Description
    Region-coupling temperature condition for conjugate heat transfer that is
    conservative and consistent when one (or both) of the coupled regions is
    a solid with a tensor conductivity whose principal axes are NOT aligned
    with the interface normal.

    It is the stock condition compressible::turbulentTemperatureRadCoupledMixed
    with two additions:

    1.  Tangential (co-normal) flux.  The stock condition closes the wall
        balance  b_own (Tw - T_own) + b_nbr (Tw - T_nbr) = qr + qrNbr  with the
        normal conductances only (turbulentTemperatureRadCoupledMixed
        FvPatchScalarField.C:435-439, 453-462, 614-616), while the Laplacian
        of an anisotropic solid adds, on the same boundary face, the explicit
        tangential part (Gamma n)_t . grad(h) of the co-normal flux
        (gaussLaplacianScheme.C:122-129, 174, 184-193).  That part reaches no
        neighbour, so the interface creates energy and a linear patch test
        fails whenever the normal is not a principal direction.  Here

            q_t = - (Gamma n)_t . grad(h)_b       (own side and neighbour side)

        is evaluated inside updateCoeffs(), i.e. from the same field state
        the solid Laplacian is about to use in the same assembly, and added
        to the wall balance of BOTH sides.  The two discrete heat rates are
        then equal and opposite in every coupling pass.

    2.  Guard.  The run is stopped (FatalError, at the first evaluation) if
        this side or the neighbour side takes its conductivity from
        'kappaMethod solidThermo' while the region's solid thermo is
        anisotropic: thermo.kappa() is then mag(kappa), not n.kappa.n
        (temperatureCoupledBase.C:328-336, constAnIsoSolidTransportI.H:66-71)
        and the two sides exchange unequal heat.

    Limitation found in the verification runs (RESULTS.md, section 3.5): the
    tangential flux is taken from the previous solve, i.e. it is explicit, and
    the coupling is only CONDITIONALLY STABLE.  In a steady-state march with a
    large pseudo-time step it diverged for anisotropy ratios of 100 and more
    at interface angles between 15 and 60 degrees, where the stock condition
    converged; it was stable in every run up to ratio 35, and at ratio 100 to
    1000 with physical heat capacities and dt = 0.05 s.  No implicit
    treatment of the tangential term is provided.

    Everything else (keywords, layer resistance, radiation fields, restart
    data) is inherited unchanged.  For two isotropic regions the condition is
    identical to the stock one.  The assembled implicit coupling
    ('useImplicit true') is not corrected and is refused.

Usage
    Both sides of the interface must use this type.
    \verbatim
    // anisotropic solid, 0/<solid>/T          // neighbour, 0/<other>/T
    type         compressible::coNormalTemperatureCoupledMixed;
    Tnbr         T;                            Tnbr        T;
    kappaMethod  directionalSolidThermo;       kappaMethod fluidThermo; // or solidThermo
    alphaAni     aniAlpha;
    value        ...;
    \endverbatim
    and in system/controlDict:   libs ("<path>/libcoNormalCoupling.so");

    Optional entry:  coNormal  true;   (false switches addition 1 off and
    leaves the guard; used for testing)
\*---------------------------------------------------------------------------*/

#ifndef Foam_coNormalTemperatureCoupledMixedFvPatchScalarField_H
#define Foam_coNormalTemperatureCoupledMixedFvPatchScalarField_H

#include "turbulentTemperatureRadCoupledMixedFvPatchScalarField.H"

namespace Foam
{
namespace compressible
{

class coNormalTemperatureCoupledMixedFvPatchScalarField
:
    public turbulentTemperatureRadCoupledMixedFvPatchScalarField
{
    typedef coNormalTemperatureCoupledMixedFvPatchScalarField this_bctype;
    typedef turbulentTemperatureRadCoupledMixedFvPatchScalarField parent_bctype;

    //- Name of the temperature field on the neighbour region
    //  (private in the parent, read again here)
    word TnbrName_;

    //- Add the tangential co-normal flux to the wall balance
    bool coNormal_;

    //- Isotropy guard already applied
    mutable bool guardChecked_;

    //- Stop if a scalar conductivity is handed over by an anisotropic solid
    void checkGuard(const fvPatch& nbrPatch) const;

public:

    TypeName("compressible::coNormalTemperatureCoupledMixed");

    coNormalTemperatureCoupledMixedFvPatchScalarField
    (
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&
    );

    coNormalTemperatureCoupledMixedFvPatchScalarField
    (
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const dictionary&
    );

    coNormalTemperatureCoupledMixedFvPatchScalarField
    (
        const this_bctype&,
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const fvPatchFieldMapper&
    );

    coNormalTemperatureCoupledMixedFvPatchScalarField
    (
        const this_bctype&,
        const DimensionedField<scalar, volMesh>&
    );

    coNormalTemperatureCoupledMixedFvPatchScalarField
    (
        const this_bctype&
    ) = delete;

    virtual tmp<fvPatchField<scalar>> clone
    (
        const DimensionedField<scalar, volMesh>& iF
    ) const
    {
        return fvPatchField<scalar>::Clone(*this, iF);
    }

    //- Update the coefficients associated with the patch field
    virtual void updateCoeffs();

    //- Write
    virtual void write(Ostream& os) const;
};

} // End namespace compressible
} // End namespace Foam

#endif
