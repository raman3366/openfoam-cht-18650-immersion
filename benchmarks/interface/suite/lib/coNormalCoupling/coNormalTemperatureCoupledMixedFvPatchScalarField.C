#include "coNormalTemperatureCoupledMixedFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "mappedPatchBase.H"
#include "solidThermo.H"
#include "coNormalFlux.H"

namespace Foam
{
namespace compressible
{

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

static void guardOneSide
(
    const fvPatch& pp,
    const fvPatchScalarField& Tp
)
{
    const auto* tcb = isA<temperatureCoupledBase>(Tp);

    if
    (
        tcb
     && tcb->KMethod() == "solidThermo"
     && coNormal::anisotropicSolid(pp.boundaryMesh().mesh())
    )
    {
        FatalErrorInFunction
            << "Patch " << pp.name() << " of region "
            << pp.boundaryMesh().mesh().name()
            << ": kappaMethod solidThermo on an ANISOTROPIC solid." << nl
            << "    The coupling would be given thermo.kappa() = mag(kappa)"
            << " instead of the normal component n.kappa.n, and the" << nl
            << "    neighbour would receive mag(kappa)/(n.kappa.n) times the"
            << " heat this region releases." << nl
            << "    Use:  kappaMethod directionalSolidThermo;"
            << "  alphaAni aniAlpha;" << nl
            << exit(FatalError);
    }
}


void coNormalTemperatureCoupledMixedFvPatchScalarField::checkGuard
(
    const fvPatch& nbrPatch
) const
{
    if (guardChecked_)
    {
        return;
    }

    guardOneSide(patch(), *this);
    guardOneSide(nbrPatch, nbrPatch.lookupPatchField<volScalarField>(TnbrName_));

    if (this->useImplicit())
    {
        FatalErrorInFunction
            << "Patch " << patch().name() << ": useImplicit is not supported"
            << " by " << typeName << exit(FatalError);
    }

    guardChecked_ = true;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

coNormalTemperatureCoupledMixedFvPatchScalarField::
coNormalTemperatureCoupledMixedFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF
)
:
    parent_bctype(p, iF),
    TnbrName_("undefined-Tnbr"),
    coNormal_(true),
    guardChecked_(false)
{}


coNormalTemperatureCoupledMixedFvPatchScalarField::
coNormalTemperatureCoupledMixedFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    parent_bctype(p, iF, dict),
    TnbrName_(dict.getOrDefault<word>("Tnbr", "T")),
    coNormal_(dict.getOrDefault<bool>("coNormal", true)),
    guardChecked_(false)
{}


coNormalTemperatureCoupledMixedFvPatchScalarField::
coNormalTemperatureCoupledMixedFvPatchScalarField
(
    const this_bctype& psf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    parent_bctype(psf, p, iF, mapper),
    TnbrName_(psf.TnbrName_),
    coNormal_(psf.coNormal_),
    guardChecked_(false)
{}


coNormalTemperatureCoupledMixedFvPatchScalarField::
coNormalTemperatureCoupledMixedFvPatchScalarField
(
    const this_bctype& psf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    parent_bctype(psf, iF),
    TnbrName_(psf.TnbrName_),
    coNormal_(psf.coNormal_),
    guardChecked_(false)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void coNormalTemperatureCoupledMixedFvPatchScalarField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const mappedPatchBase& mpp =
        mappedPatchFieldBase<scalar>::mapper(patch(), this->internalField());

    if (!mpp.sameWorld())
    {
        FatalErrorInFunction
            << typeName << " does not support multi-world coupling"
            << exit(FatalError);
    }

    const fvPatch& nbrPatch =
        refCast<const fvMesh>(mpp.sampleMesh()).boundary()
        [
            mpp.samplePolyPatch().index()
        ];

    checkGuard(nbrPatch);

    // Tangential part of the co-normal flux of each side, as a source of the
    // wall balance (heat arriving at the wall is minus the flux into the
    // region).  Zero for a region that is not an anisotropic solid.
    scalarField qTan(this->size(), Zero);

    if (coNormal_)
    {
        qTan = -coNormal::tangentialFlux(patch());

        scalarField qTanNbr(-coNormal::tangentialFlux(nbrPatch));
        mappedPatchFieldBase<scalar>::distribute
        (
            this->internalField().name() + "_qTan",
            qTanNbr
        );

        qTan += qTanNbr;
    }

    // Conductivity the parent is about to use for refGrad (the patch value
    // is not changed by updateCoeffs)
    const scalarField kappaTp(this->kappa(*this));

    // Stock coefficients: valueFraction, refValue = neighbour cell
    // temperature, refGrad = (qr + qrNbr)/kappa
    parent_bctype::updateCoeffs();

    this->refGrad() += qTan/kappaTp;
}


void coNormalTemperatureCoupledMixedFvPatchScalarField::write
(
    Ostream& os
) const
{
    parent_bctype::write(os);
    os.writeEntryIfDifferent<bool>("coNormal", true, coNormal_);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makePatchTypeField
(
    fvPatchScalarField,
    coNormalTemperatureCoupledMixedFvPatchScalarField
);

} // End namespace compressible
} // End namespace Foam
