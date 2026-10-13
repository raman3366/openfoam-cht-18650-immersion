#include "coNormalFlux.H"
#include "solidThermo.H"
#include "fvc.H"
#include "volFields.H"

bool Foam::coNormal::anisotropicSolid(const fvMesh& mesh)
{
    const auto* thermoPtr =
        mesh.findObject<solidThermo>(basicThermo::dictName);

    return (thermoPtr && !thermoPtr->isotropic());
}


Foam::tmp<Foam::scalarField>
Foam::coNormal::tangentialFlux(const fvPatch& pp)
{
    auto tq = tmp<scalarField>::New(pp.size(), Zero);

    const fvMesh& mesh = pp.boundaryMesh().mesh();

    if (!anisotropicSolid(mesh) || !pp.size())
    {
        return tq;
    }

    const auto& thermo =
        mesh.lookupObject<solidThermo>(basicThermo::dictName);

    const label patchi = pp.index();

    // Same gradient object the tensor Laplacian builds for its explicit part
    const volVectorField gradh(fvc::grad(thermo.he()));

    const vectorField n(pp.nf());
    const vectorField Gn(n & thermo.aniAlpha().boundaryField()[patchi]);

    tq.ref() = ((Gn - (Gn & n)*n) & gradh.boundaryField()[patchi]);

    return tq;
}


Foam::tmp<Foam::scalarField>
Foam::coNormal::normalFlux(const fvPatch& pp)
{
    const fvMesh& mesh = pp.boundaryMesh().mesh();
    const label patchi = pp.index();

    const auto& thermo =
        mesh.lookupObject<solidThermo>(basicThermo::dictName);

    const scalarField snGradh(thermo.he().boundaryField()[patchi].snGrad());

    if (thermo.isotropic())
    {
        return thermo.alpha().boundaryField()[patchi]*snGradh;
    }

    const vectorField n(pp.nf());

    return ((n & thermo.aniAlpha().boundaryField()[patchi]) & n)*snGradh;
}
