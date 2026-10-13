/*---------------------------------------------------------------------------*\
  anisotropicCouplingGuard -- function object (OpenFOAM v2606, ESI)

Description
    Looks at every region of the case.  For each region whose solid thermo is
    anisotropic (solidThermo::isotropic() false; solidThermo.H:170-171) it
    examines the temperature boundary field and reports every patch whose
    condition derives from temperatureCoupledBase and takes its conductivity
    from 'kappaMethod solidThermo' (temperatureCoupledBase.C:328-336): there
    thermo.kappa() is mag(kappa) (constAnIsoSolidTransportI.H:66-71) and the
    neighbour receives mag(kappa)/(n.kappa.n) times the heat the solid
    releases.  It works with the stock coupled conditions; nothing has to be
    recompiled.

    It also names the regions for which the stock wallHeatFlux object is not a
    heat flux (it multiplies snGrad(h) by the scalar thermo.alpha();
    wallHeatFlux_wall.cxx:87).

Usage (system/controlDict, one entry for the whole case)
    \verbatim
    guard
    {
        type    anisotropicCouplingGuard;
        libs    ("<path>/libcoNormalCoupling.so");
        action  error;      // error (default) | warn
    }
    \endverbatim

    With 'action error' the run stops when the function objects are loaded,
    before the first time step.  OpenFOAM downgrades a FatalError raised in a
    function-object constructor to a warning and discards the object unless
    the entry carries 'errors strict' (functionObject.H:141-150;
    functionObjectList.C:1108-1185), but treats a FatalIOError as fatal by
    default.  The condition is an input error of the temperature dictionary,
    so it is raised as a FatalIOError naming that file.  Only an explicit
    'errors warn' or 'errors ignore' on the entry turns it into a warning.
\*---------------------------------------------------------------------------*/

#ifndef functionObjects_anisotropicCouplingGuard_H
#define functionObjects_anisotropicCouplingGuard_H

#include "timeFunctionObject.H"

namespace Foam
{
namespace functionObjects
{

class anisotropicCouplingGuard
:
    public timeFunctionObject
{
    //- Stop (true) or warn (false)
    bool fatal_;

    //- A check that saw at least one thermo object has been made
    bool checked_;

    //- Number of offending patches found by the last check
    label nBad_;

    //- Examine all regions; returns false if no thermo was found yet.
    //  raise: issue the FatalError now (otherwise only a warning)
    bool check(const bool raise);

public:

    TypeName("anisotropicCouplingGuard");

    anisotropicCouplingGuard
    (
        const word& name,
        const Time& runTime,
        const dictionary& dict
    );

    virtual ~anisotropicCouplingGuard() = default;

    virtual bool read(const dictionary& dict);
    virtual bool execute();
    virtual bool write();
};

} // End namespace functionObjects
} // End namespace Foam

#endif
