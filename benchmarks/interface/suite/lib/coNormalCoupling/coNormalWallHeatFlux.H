/*---------------------------------------------------------------------------*\
  coNormalWallHeatFlux -- function object (OpenFOAM v2606, ESI)

Description
    Wall heat flux of a SOLID region, evaluated with the operator the solver
    itself applies on the patch (coNormalFlux.H):

        q = (n.Gamma.n) snGrad(h) + (Gamma n)_t . grad(h)_b      [W/m2]

    positive into the region; Gamma = aniAlpha for an anisotropic solid and
    the scalar alpha for an isotropic one (then the second term is zero and
    the result equals the stock wallHeatFlux object).  The stock object uses
    the scalar thermo.alpha() = mag(kappa)/Cp on anisotropic solids
    (wallHeatFlux_wall.cxx:87; heSolidThermo.C:65-68) and cannot be used for
    an energy audit there.

    Writes the volScalarField <name of the object> (boundary values) at write
    time and reports, per patch, the integral of the normal part, of the
    tangential part and of their sum [W].

Usage
    \verbatim
    qSolid
    {
        type     coNormalWallHeatFlux;
        libs     ("<path>/libcoNormalCoupling.so");
        region   <solid region>;
        patches  (".*");          // optional, default: all non-empty patches
    }
    \endverbatim
\*---------------------------------------------------------------------------*/

#ifndef functionObjects_coNormalWallHeatFlux_H
#define functionObjects_coNormalWallHeatFlux_H

#include "fvMeshFunctionObject.H"
#include "writeFile.H"
#include "volFields.H"
#include "wordRes.H"

namespace Foam
{
namespace functionObjects
{

class coNormalWallHeatFlux
:
    public fvMeshFunctionObject,
    public writeFile
{
    //- Selected patches
    labelList patchIDs_;

    //- Patch selection
    wordRes patchNames_;

    //- Output field
    volScalarField& field();

    virtual void writeFileHeader(Ostream& os) const;

public:

    TypeName("coNormalWallHeatFlux");

    coNormalWallHeatFlux
    (
        const word& name,
        const Time& runTime,
        const dictionary& dict
    );

    virtual ~coNormalWallHeatFlux() = default;

    virtual bool read(const dictionary& dict);
    virtual bool execute();
    virtual bool write();
};

} // End namespace functionObjects
} // End namespace Foam

#endif
