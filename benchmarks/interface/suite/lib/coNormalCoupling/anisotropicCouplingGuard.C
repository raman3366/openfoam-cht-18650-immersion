#include "anisotropicCouplingGuard.H"
#include "addToRunTimeSelectionTable.H"
#include "fvMesh.H"
#include "volFields.H"
#include "solidThermo.H"
#include "temperatureCoupledBase.H"
#include "Time.H"

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(anisotropicCouplingGuard, 0);
    addToRunTimeSelectionTable
    (
        functionObject,
        anisotropicCouplingGuard,
        dictionary
    );
}
}


bool Foam::functionObjects::anisotropicCouplingGuard::check(const bool raise)
{
    bool sawThermo = false;
    nBad_ = 0;
    OStringStream msg;
    fileName firstFile;

    for (const fvMesh& mesh : time_.csorted<fvMesh>())
    {
        const auto* thermoPtr =
            mesh.findObject<solidThermo>(basicThermo::dictName);

        if (!thermoPtr)
        {
            continue;
        }
        sawThermo = true;

        if (thermoPtr->isotropic())
        {
            continue;
        }

        Info<< type() << " " << name() << ": region " << mesh.name()
            << " is an anisotropic solid; the stock wallHeatFlux object"
            << " reports mag(kappa)/Cp*snGrad(h) there, which is not the"
            << " heat flux (use coNormalWallHeatFlux)" << endl;

        const auto& Tbf = thermoPtr->T().boundaryField();

        forAll(Tbf, patchi)
        {
            const auto* tcb = isA<temperatureCoupledBase>(Tbf[patchi]);

            if (tcb && tcb->KMethod() == "solidThermo")
            {
                if (!nBad_)
                {
                    firstFile = thermoPtr->T().objectPath();
                }
                ++nBad_;
                msg << "    region " << mesh.name() << ", patch "
                    << mesh.boundary()[patchi].name() << " ("
                    << Tbf[patchi].type() << ")" << nl;
            }
        }
    }

    if (nBad_)
    {
        const string text
        (
            "kappaMethod solidThermo on an anisotropic solid:\n"
          + msg.str()
          + "    The coupling is given mag(kappa) instead of n.kappa.n; the"
            " neighbour receives\n    mag(kappa)/(n.kappa.n) times the heat"
            " the solid releases.\n    Use:  kappaMethod"
            " directionalSolidThermo;  alphaAni aniAlpha;\n"
        );

        if (fatal_ && raise)
        {
            // An input error of the temperature dictionary: raised as a
            // FatalIOError, which the function-object framework treats as
            // fatal on construction unless the entry says errors warn/ignore
            // (functionObjectList.C:1134-1172)
            FatalIOError
            (
                FUNCTION_NAME, __FILE__, __LINE__, firstFile, 0, 0
            )   << text.c_str() << exit(FatalIOError);
        }
        else
        {
            WarningInFunction << text.c_str() << endl;
        }
    }

    return sawThermo;
}


Foam::functionObjects::anisotropicCouplingGuard::anisotropicCouplingGuard
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    timeFunctionObject(name, runTime),
    fatal_(true),
    checked_(false),
    nBad_(0)
{
    read(dict);

    checked_ = check(true);
}


bool Foam::functionObjects::anisotropicCouplingGuard::read
(
    const dictionary& dict
)
{
    functionObject::read(dict);

    const word action(dict.getOrDefault<word>("action", "error"));

    if (action != "error" && action != "warn")
    {
        FatalIOErrorInFunction(dict)
            << "action must be 'error' or 'warn', not " << action
            << exit(FatalIOError);
    }
    fatal_ = (action == "error");

    return true;
}


bool Foam::functionObjects::anisotropicCouplingGuard::execute()
{
    // Only reached with an offending patch if no thermo existed yet when the
    // object was constructed
    if (!checked_ || (fatal_ && nBad_))
    {
        checked_ = check(true);
    }
    return true;
}


bool Foam::functionObjects::anisotropicCouplingGuard::write()
{
    return true;
}
