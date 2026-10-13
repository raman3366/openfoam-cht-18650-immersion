#!/usr/bin/env python3
r"""
blockMeshDict generator: 18650 cell (quarter model) centred in a cuboid
dielectric bath.  Pure-hex O-grid -- no snappyHexMesh, so the mesh is
deterministic and the cell/fluid interface is body-fitted and perfectly conformal.

Quarter symmetry: the cell axis sits on the (x=0, y=0) edge of the domain, so
x=0 and y=0 are symmetry planes and we model one quadrant of the bath.

xy topology (quarter O-grid), 5 columns of blocks:

      y=L  9--------------8      F2 : outer fluid, NE-to-N sector
           |  F2         /|      F1 : outer fluid, E-to-NE sector
      y=R  6\           / |      S2 : jellyroll annulus, N sector
           |  \5______ /  |      S1 : jellyroll annulus, E sector
      y=a  3---2      \   |      S0 : jellyroll square core
           |S0 |S1   \ F1 |
           0---1---4------7
             a   R        L                x

Stacked in z as three layers: fluid below the cell, cell+fluid beside it,
fluid above.  Columns S0/S1/S2 in the middle layer are the battery; everything
else is fluid.  Vertical block faces match 1:1 so blockMesh merges them.
r"""
import argparse, math, sys

p = argparse.ArgumentParser()
p.add_argument("--nc",   type=int,   default=16, help="cells circumferentially per 45 deg sector (= core block edge)")
p.add_argument("--nrs",  type=int,   default=10, help="radial cells in the jellyroll annulus")
p.add_argument("--nrf",  type=int,   default=40, help="radial cells in the fluid, cell wall -> box wall")
p.add_argument("--nz1",  type=int,   default=28, help="axial cells below the cell")
p.add_argument("--nz2",  type=int,   default=65, help="axial cells over the cell height")
p.add_argument("--nz3",  type=int,   default=28, help="axial cells above the cell")
p.add_argument("--grf",  type=float, default=5.0, help="fluid radial expansion ratio (last/first cell)")
p.add_argument("--out",  default="blockMeshDict")
a_ = p.parse_args()

# ---- geometry, mm --------------------------------------------------------
R   = 9.0        # 18650 radius
HC  = 65.0       # 18650 height
L   = 30.0       # bath half-width  (60 x 60 mm internal footprint)
H   = 130.0      # bath internal height
A   = 4.5        # O-grid core half-width
D   = R / math.sqrt(2.0)
ZB  = (H - HC) / 2.0     # 32.5  cell base
ZT  = ZB + HC            # 97.5  cell top
ZS  = [0.0, ZB, ZT, H]

# arc interpolation points at 22.5 and 67.5 degrees
AX, AY = R * math.cos(math.radians(22.5)), R * math.sin(math.radians(22.5))

XY = [(0,0), (A,0), (A,A), (0,A), (R,0), (D,D), (0,R), (L,0), (L,L), (0,L)]

def v(level, i):
    return 10 * level + i

verts = []
for lev, z in enumerate(ZS):
    for i, (x, y) in enumerate(XY):
        verts.append("    ({:.6f} {:.6f} {:.6f})   // {:d}".format(x, y, z, v(lev, i)))

# ---- blocks --------------------------------------------------------------
# (name, v0..v3 in the xy plane, (n_localx, n_localy), grading string)
COLS = [
    ("S0", (0, 1, 2, 3), (a_.nc,  a_.nc ), "simpleGrading (1 1 1)"),
    ("S1", (1, 4, 5, 2), (a_.nrs, a_.nc ), "simpleGrading (1 1 1)"),
    ("S2", (3, 2, 5, 6), (a_.nc,  a_.nrs), "simpleGrading (1 1 1)"),
    ("F1", (4, 7, 8, 5), (a_.nrf, a_.nc ), "simpleGrading ({:g} 1 1)".format(a_.grf)),
    ("F2", (6, 5, 8, 9), (a_.nc,  a_.nrf), "simpleGrading (1 {:g} 1)".format(a_.grf)),
]
LAYERS = [(0, a_.nz1, "below cell"), (1, a_.nz2, "cell height"), (2, a_.nz3, "above cell")]

blocks = []
for lev, nz, tag in LAYERS:
    blocks.append("    // ---- layer {:d}: z = {:g} .. {:g} mm  ({}) ----".format(lev + 1, ZS[lev], ZS[lev + 1], tag))
    for name, q, (ni, nj), grad in COLS:
        ids = [v(lev, k) for k in q] + [v(lev + 1, k) for k in q]
        blocks.append("    hex ({}) ({:d} {:d} {:d}) {}   // {}{:d}".format(
            " ".join(str(i) for i in ids), ni, nj, nz, grad, name, lev + 1))

edges = []
for lev, z in enumerate(ZS):
    edges.append("    arc {:d} {:d} ({:.6f} {:.6f} {:.6f})".format(v(lev,4), v(lev,5), AX, AY, z))
    edges.append("    arc {:d} {:d} ({:.6f} {:.6f} {:.6f})".format(v(lev,5), v(lev,6), AY, AX, z))

# ---- boundary ------------------------------------------------------------
def face(lev, quad):
    """quad given as xy indices at (lower, lower, upper, upper) -> face string"""
    i, j = quad
    return "            ({:d} {:d} {:d} {:d})".format(v(lev,i), v(lev,j), v(lev+1,j), v(lev+1,i))

symX_edges = [(0,3), (3,6), (6,9)]      # x = 0 : S0, S2, F2
symY_edges = [(0,1), (1,4), (4,7)]      # y = 0 : S0, S1, F1
wallX_edges = [(7,8)]                   # x = L : F1
wallY_edges = [(8,9)]                   # y = L : F2  (ordered for outward normal)

def side(edgelist):
    return "\n".join(face(lev, e) for lev in (0,1,2) for e in edgelist)

BOT = [(0,1,2,3), (1,4,5,2), (3,2,5,6), (4,7,8,5), (6,5,8,9)]
bottom = "\n".join("            ({:d} {:d} {:d} {:d})".format(*[v(0,k) for k in q]) for q in BOT)
top    = "\n".join("            ({:d} {:d} {:d} {:d})".format(*[v(3,k) for k in q]) for q in BOT)

TXT = """/*--------------------------------*- C++ -*----------------------------------*\\
| 18650 cell immersed in a dielectric ester bath -- quarter model            |
| Generated by tools/make_blockMesh.py -- DO NOT EDIT BY HAND                |
| cell  D {D:g} mm x H {HC:g} mm, axis on the (x=0,y=0) symmetry edge          |
| bath  {W:g} x {W:g} x {H:g} mm internal (quarter modelled: {L:g} x {L:g} x {H:g})    |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale   0.001;   // dimensions above are millimetres

vertices
(
{verts}
);

blocks
(
{blocks}
);

edges
(
{edges}
);

boundary
(
    symmetryX
    {{
        type            symmetryPlane;
        faces
        (
{symX}
        );
    }}

    symmetryY
    {{
        type            symmetryPlane;
        faces
        (
{symY}
        );
    }}

    bathWallX
    {{
        type            wall;
        faces
        (
{wallX}
        );
    }}

    bathWallY
    {{
        type            wall;
        faces
        (
{wallY}
        );
    }}

    bathFloor
    {{
        type            wall;
        faces
        (
{bottom}
        );
    }}

    bathLid
    {{
        type            wall;
        faces
        (
{top}
        );
    }}
);

mergePatchPairs ();

// ************************************************************************* //
r"""

out = TXT.format(
    D=2*R, HC=HC, W=2*L, H=H, L=L,
    verts="\n".join(verts), blocks="\n".join(blocks), edges="\n".join(edges),
    symX=side(symX_edges), symY=side(symY_edges),
    wallX=side(wallX_edges), wallY=side(wallY_edges),
    bottom=bottom, top=top,
)
with open(a_.out, "w") as f:
    f.write(out)

# ---- report --------------------------------------------------------------
xy_solid = a_.nc*a_.nc + a_.nrs*a_.nc + a_.nc*a_.nrs
xy_fluid = 2 * a_.nrf * a_.nc
nfl = (xy_solid + xy_fluid)*(a_.nz1 + a_.nz3) + xy_fluid*a_.nz2
nso = xy_solid * a_.nz2
r = a_.grf ** (1.0/(a_.nrf - 1))
first = (L - R) * (r - 1.0) / (r**a_.nrf - 1.0)
print("wrote {}".format(a_.out))
print("  cells: fluid {:,}  battery {:,}  total {:,}".format(nfl, nso, nfl+nso))
print("  fluid 1st cell off the can: {:.3f} mm   last: {:.3f} mm".format(first, first*a_.grf))
print("  axial cell size over the can: {:.3f} mm   annulus radial: {:.3f} mm"
      .format(HC/a_.nz2, (R - A*math.sqrt(2))/a_.nrs))
