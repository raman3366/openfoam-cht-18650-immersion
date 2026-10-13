#include "coNormalInterfaceFlux.H"
#include "addToRunTimeSelectionTable.H"
#include "fvc.H"
#include "solidThermo.H"
#include "mappedPatchBase.H"
#include "temperatureCoupledBase.H"

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(coNormalInterfaceFlux, 0);
    addToRunTimeSelectionTable
    (
        functionObject,
        coNormalInterfaceFlux,
        dictionary
    );
}
}


void Foam::functionObjects::coNormalInterfaceFlux::checkGuard() const
{
    const auto& thermo =
        mesh_.lookupObject<solidThermo>(basicThermo::dictName);

    if (thermo.isotropic())
    {
        return;
    }

    const auto& Tbf = thermo.T().boundaryField();

    forAll(Tbf, patchi)
    {
        const auto* tcb = isA<temperatureCoupledBase>(Tbf[patchi]);

        if (tcb && tcb->KMethod() == "solidThermo")
        {
            FatalErrorInFunction
                << "Patch " << mesh_.boundary()[patchi].name()
                << " of region " << mesh_.name()
                << " uses kappaMethod solidThermo, but the solid thermo is"
                << " anisotropic: the coupling would use mag(kappa) instead"
                << " of n.kappa.n and the two sides of the interface would"
                << " exchange unequal heat." << nl
                << "    Use kappaMethod directionalSolidThermo with"
                << " alphaAni aniAlpha." << nl
                << exit(FatalError);
        }
    }
}


void Foam::functionObjects::coNormalInterfaceFlux::update()
{
    auto& qTan = mesh_.lookupObjectRef<volScalarField>(fieldName_);

    const auto& thermo =
        mesh_.lookupObject<solidThermo>(basicThermo::dictName);

    if (thermo.isotropic())
    {
        return;
    }

    const volScalarField& h = thermo.he();
    const volSymmTensorField& aniAlpha = thermo.aniAlpha();

    // Same gradient the tensor Laplacian uses for its explicit part
    // (gaussLaplacianScheme.C:122-129): boundary value = adjacent cell
    // value with the normal component replaced (gaussGrad.C:205-233)
    const volVectorField gradh(fvc::grad(h));

    auto& qbf = qTan.boundaryFieldRef();

    forAll(qbf, patchi)
    {
        const fvPatch& pp = mesh_.boundary()[patchi];

        if (!isA<mappedPatchBase>(pp.patch()))
        {
            continue;
        }

        const vectorField n(pp.nf());
        const vectorField Gn(n & aniAlpha.boundaryField()[patchi]);
        const vectorField Gt(Gn - (Gn & n)*n);

        qbf[patchi] = -(Gt & gradh.boundaryField()[patchi]);
    }
}


Foam::functionObjects::coNormalInterfaceFlux::coNormalInterfaceFlux
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    fieldName_("qTan"),
    guard_(true)
{
    read(dict);

    if (!mesh_.foundObject<volScalarField>(fieldName_))
    {
        auto* ptr = new volScalarField
        (
            IOobject
            (
                fieldName_,
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh_,
            dimensionedScalar(dimMass/pow3(dimTime), Zero)
        );
        mesh_.objectRegistry::store(ptr);
    }

    if (guard_)
    {
        checkGuard();
    }

    update();
}


bool Foam::functionObjects::coNormalInterfaceFlux::read(const dictionary& dict)
{
    fvMeshFunctionObject::read(dict);
    fieldName_ = dict.getOrDefault<word>("field", "qTan");
    guard_ = dict.getOrDefault<bool>("guard", true);
    return true;
}


bool Foam::functionObjects::coNormalInterfaceFlux::execute()
{
    update();
    return true;
}


bool Foam::functionObjects::coNormalInterfaceFlux::write()
{
    return true;
}
