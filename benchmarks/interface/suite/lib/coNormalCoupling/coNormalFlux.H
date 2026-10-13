/*---------------------------------------------------------------------------*\
  coNormalFlux -- helpers shared by the boundary condition and the function
  objects of libcoNormalCoupling (OpenFOAM v2606, ESI).

  For a solid region whose thermo is anisotropic (solidThermo::isotropic() is
  false) the energy equation is  ddt(rho h) - div(Gamma . grad h) = S  with
  Gamma = aniAlpha = kappa/Cp rotated into global axes
  (solidThermo.C:299-306, heSolidThermo.C:145-191).  The heat flux INTO the
  region through a boundary face with outward unit normal n is the co-normal
  derivative

      q = n.Gamma.grad(h) = (n.Gamma.n) snGrad(h)  +  (Gamma n)_t . grad(h)_b
          `------ normal part -------'   `----- tangential part -----'

  with (Gamma n)_t = Gamma n - (n.Gamma.n) n.  The Gauss Laplacian treats the
  first part implicitly and the second explicitly, on boundary faces as well
  (gaussLaplacianScheme.C:122-129, 167-193); grad(h)_b is the patch value of
  fvc::grad(h), i.e. the adjacent cell gradient with its normal component
  replaced (gaussGrad.C:205-233).  The functions below evaluate exactly these
  two parts.  betav (porosity-like factor of solveSolid.H) is taken as 1.
\*---------------------------------------------------------------------------*/

#ifndef coNormalFlux_H
#define coNormalFlux_H

#include "fvMesh.H"
#include "fvPatch.H"
#include "scalarField.H"

namespace Foam
{
namespace coNormal
{

//- True if the region holds a solidThermo that is not isotropic
bool anisotropicSolid(const fvMesh& mesh);

//- Tangential part of the co-normal flux into the region [W/m2];
//- zero for regions that are not anisotropic solids
tmp<scalarField> tangentialFlux(const fvPatch& pp);

//- Normal part (n.Gamma.n) snGrad(h) of the flux into the region [W/m2];
//- alpha*snGrad(h) with the scalar alpha for isotropic solids
tmp<scalarField> normalFlux(const fvPatch& pp);

} // End namespace coNormal
} // End namespace Foam

#endif
