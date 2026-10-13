#include "coNormalWallHeatFlux.H"
#include "addToRunTimeSelectionTable.H"
#include "emptyPolyPatch.H"
#include "solidThermo.H"
#include "coNormalFlux.H"

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(coNormalWallHeatFlux, 0);
    addToRunTimeSelectionTable
    (
        functionObject,
        coNormalWallHeatFlux,
        dictionary
    );
}
}


Foam::volScalarField& Foam::functionObjects::coNormalWallHeatFlux::field()
{
    auto* ptr = mesh_.getObjectPtr<volScalarField>(name());

    if (!ptr)
    {
        ptr = new volScalarField
        (
            IOobject
            (
                name(),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                IOobject::REGISTER
            ),
            mesh_,
            dimensionedScalar(dimMass/pow3(dimTime), Zero)
        );
        mesh_.objectRegistry::store(ptr);
    }

    return *ptr;
}


void Foam::functionObjects::coNormalWallHeatFlux::writeFileHeader
(
    Ostream& os
) const
{
    writeHeader(os, "Co-normal wall heat flux (positive into the region)");
    writeCommented(os, "Time");
    writeTabbed(os, "patch");
    writeTabbed(os, "normal[W]");
    writeTabbed(os, "tangential[W]");
    writeTabbed(os, "total[W]");
    os  << endl;
}


Foam::functionObjects::coNormalWallHeatFlux::coNormalWallHeatFlux
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    writeFile(obr_, name, typeName, dict)
{
    read(dict);
    writeFileHeader(file());
    field();
}


bool Foam::functionObjects::coNormalWallHeatFlux::read(const dictionary& dict)
{
    fvMeshFunctionObject::read(dict);
    writeFile::read(dict);

    patchNames_.clear();
    dict.readIfPresent("patches", patchNames_);

    DynamicList<label> ids;
    const polyBoundaryMesh& pbm = mesh_.boundaryMesh();

    forAll(pbm, patchi)
    {
        if (isA<emptyPolyPatch>(pbm[patchi]) || pbm[patchi].coupled())
        {
            continue;
        }
        if (patchNames_.empty() || patchNames_.match(pbm[patchi].name()))
        {
            ids.append(patchi);
        }
    }
    patchIDs_.transfer(ids);

    return true;
}


bool Foam::functionObjects::coNormalWallHeatFlux::execute()
{
    if (!mesh_.foundObject<solidThermo>(basicThermo::dictName))
    {
        FatalErrorInFunction
            << "Region " << mesh_.name() << " has no solidThermo; use the"
            << " stock wallHeatFlux object for fluid regions"
            << exit(FatalError);
    }

    volScalarField& q = field();
    auto& qbf = q.boundaryFieldRef();

    Log << type() << " " << name() << " write:" << nl;

    for (const label patchi : patchIDs_)
    {
        const fvPatch& pp = mesh_.boundary()[patchi];

        const scalarField qn(coNormal::normalFlux(pp));
        const scalarField qt(coNormal::tangentialFlux(pp));

        qbf[patchi] = qn + qt;

        const scalar Qn = gSum(pp.magSf()*qn);
        const scalar Qt = gSum(pp.magSf()*qt);

        if (Pstream::master())
        {
            writeCurrentTime(file());
            file()
                << token::TAB << pp.name()
                << token::TAB << Qn
                << token::TAB << Qt
                << token::TAB << Qn + Qt
                << endl;
        }

        Log << "    " << pp.name() << ": normal " << Qn << " W, tangential "
            << Qt << " W, total " << Qn + Qt << " W" << nl;
    }
    Log << endl;

    return true;
}


bool Foam::functionObjects::coNormalWallHeatFlux::write()
{
    field().write();
    return true;
}
