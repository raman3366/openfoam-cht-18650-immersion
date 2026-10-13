/*---------------------------------------------------------------------------*\
  coNormalInterfaceFlux -- function object for OpenFOAM v2606 (ESI)

  Purpose (see DERIVATION.md, Sections C and G)
  1. Guard: stop the run if a temperature-coupled patch of an anisotropic
     solid takes its conductivity from 'kappaMethod solidThermo', which
     returns the magnitude of the principal-conductivity vector instead of
     the normal component n.K.n (solidThermo::isotropic() is the test).
  2. Correction: maintain, on the coupled patches of the anisotropic region,
     the tangential part of the co-normal flux

         qTan = - (n.Gamma - (n.Gamma.n) n) . grad(h)        [W/m2]

     with Gamma = aniAlpha (kappa/Cp) and grad(h) evaluated exactly as the
     solid Laplacian evaluates them on that face.  The stock coupled
     condition compressible::turbulentTemperatureRadCoupledMixed adds
     (qr + qrNbr) to the interface balance, so entering

         solid side:   qr     qTan;   qrNbr  none;
         other side:   qr     none;   qrNbr  qTan;

     makes the wall temperature satisfy the full co-normal flux balance and
     the two discrete fluxes equal.  The term is explicit (one time step, or
     one coupling pass, behind), like the solid's own off-normal term.

  Usage (system/controlDict)
      coNormal
      {
          type    coNormalInterfaceFlux;
          libs    ("<path>/libcoNormalInterface.so");
          region  <anisotropic solid region>;
          field   qTan;        // optional, default qTan
          guard   true;        // optional, default true
      }
\*---------------------------------------------------------------------------*/

#ifndef functionObjects_coNormalInterfaceFlux_H
#define functionObjects_coNormalInterfaceFlux_H

#include "fvMeshFunctionObject.H"
#include "volFields.H"

namespace Foam
{
namespace functionObjects
{

class coNormalInterfaceFlux
:
    public fvMeshFunctionObject
{
    //- Name of the flux field
    word fieldName_;

    //- Apply the isotropy guard
    bool guard_;

    //- Recompute the patch values of the flux field
    void update();

    //- Abort on scalar conductivity used with an anisotropic solid
    void checkGuard() const;

public:

    TypeName("coNormalInterfaceFlux");

    coNormalInterfaceFlux
    (
        const word& name,
        const Time& runTime,
        const dictionary& dict
    );

    virtual ~coNormalInterfaceFlux() = default;

    virtual bool read(const dictionary& dict);
    virtual bool execute();
    virtual bool write();
};

} // End namespace functionObjects
} // End namespace Foam

#endif
