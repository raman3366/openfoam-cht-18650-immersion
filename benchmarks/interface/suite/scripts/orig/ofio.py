#!/usr/bin/env python3
"""
ofio.py -- minimal ASCII OpenFOAM mesh and field reader used by gencase.py and
post.py.  Geometry (face centres, face area vectors, cell centres, cell volumes)
is recomputed from points/faces/owner/neighbour with the same pyramid/triangle
decomposition OpenFOAM uses, so nothing depends on a utility being able to load
the coupled boundary conditions.  post.py cross-checks the result against the
solver's own `C` and `V` fields when they are present.
"""
import os
import re
import numpy as np

_COMMENT = re.compile(r'//[^\n]*|/\*.*?\*/', re.S)
_NUM = r'[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|nan|inf)'


def _body(path):
    txt = _COMMENT.sub('', open(path).read())
    m = re.search(r'FoamFile\s*\{[^}]*\}', txt)
    return txt[m.end():] if m else txt


def _matching(txt, i, op='(', cl=')'):
    """index just past the bracket matching txt[i] == op"""
    depth = 0
    for j in range(i, len(txt)):
        ch = txt[j]
        if ch == op:
            depth += 1
        elif ch == cl:
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError('unbalanced brackets')


def _numbers(s):
    return np.array(s.replace('(', ' ').replace(')', ' ').split(), dtype=float)


# ----------------------------------------------------------------------- mesh


class Mesh:
    def __init__(self, polyMeshDir):
        d = polyMeshDir
        b = _body(os.path.join(d, 'points'))
        m = re.search(r'(\d+)\s*\(', b)
        n = int(m.group(1))
        self.points = _numbers(b[m.end() - 1:_matching(b, m.end() - 1)]).reshape(n, 3)

        b = _body(os.path.join(d, 'faces'))
        m = re.search(r'(\d+)\s*\(', b)
        nf = int(m.group(1))
        inner = b[m.end():]
        fl = re.findall(r'(\d+)\(([^)]*)\)', inner)
        assert len(fl) == nf, (len(fl), nf)
        self.faces = [np.array(v.split(), dtype=int) for _, v in fl]

        def labels(name):
            b = _body(os.path.join(d, name))
            m = re.search(r'(\d+)\s*\(', b)
            n = int(m.group(1))
            return np.array(b[m.end():_matching(b, m.end() - 1) - 1].split(), dtype=int).reshape(n)

        self.owner = labels('owner')
        self.neighbour = labels('neighbour')
        self.nInternal = len(self.neighbour)
        self.nCells = int(self.owner.max()) + 1

        b = _body(os.path.join(d, 'boundary'))
        self.patches = {}
        for name, blk in re.findall(r'([A-Za-z_][\w.:-]*)\s*\{([^{}]*)\}', b):
            g = dict(re.findall(r'(\w+)\s+([^;]+);', blk))
            if 'nFaces' in g:
                self.patches[name] = dict(type=g.get('type', '').strip(), n=int(g['nFaces']),
                                          start=int(g['startFace']),
                                          sampleRegion=g.get('sampleRegion', '').strip(),
                                          samplePatch=g.get('samplePatch', '').strip())
        self._geometry()

    def _geometry(self):
        nf = len(self.faces)
        Cf, Sf = np.zeros((nf, 3)), np.zeros((nf, 3))
        P = self.points
        for f, v in enumerate(self.faces):       # primitiveMeshFaceCentresAndAreas
            p = P[v]
            if len(v) == 3:
                Cf[f] = p.mean(axis=0)
                Sf[f] = 0.5 * np.cross(p[1] - p[0], p[2] - p[0])
                continue
            fc = p.mean(axis=0)
            nxt = np.roll(p, -1, axis=0)
            nrm = np.cross(nxt - p, fc - p)
            a = np.linalg.norm(nrm, axis=1)
            c = p + nxt + fc
            Sf[f] = 0.5 * nrm.sum(axis=0)
            Cf[f] = (a[:, None] * c).sum(axis=0) / (3.0 * a.sum()) if a.sum() > 0 else fc
        self.Cf, self.Sf = Cf, Sf
        self.magSf = np.linalg.norm(Sf, axis=1)
        nc, ni = self.nCells, self.nInternal
        own, nei = self.owner, self.neighbour
        cEst, cnt = np.zeros((nc, 3)), np.zeros(nc)        # primitiveMeshCellCentresAndVols
        np.add.at(cEst, own, Cf); np.add.at(cnt, own, 1)
        np.add.at(cEst, nei, Cf[:ni]); np.add.at(cnt, nei, 1)
        cEst /= cnt[:, None]
        C, V = np.zeros((nc, 3)), np.zeros(nc)
        pv = np.einsum('ij,ij->i', Sf, Cf - cEst[own])
        np.add.at(C, own, pv[:, None] * (0.75 * Cf + 0.25 * cEst[own])); np.add.at(V, own, pv)
        pv = np.einsum('ij,ij->i', Sf[:ni], cEst[nei] - Cf[:ni])
        np.add.at(C, nei, pv[:, None] * (0.75 * Cf[:ni] + 0.25 * cEst[nei])); np.add.at(V, nei, pv)
        self.C = C / V[:, None]
        self.V = V / 3.0

    def patch(self, name):
        """Geometry of a boundary patch: owner cells, face centres, unit normals,
        areas and deltaCoeffs = 1/|n.(Cf - C_owner)| (fvPatch.C:156-161,
        basicFvGeometryScheme.C:197-200)."""
        p = self.patches[name]
        sl = slice(p['start'], p['start'] + p['n'])
        own = self.owner[sl]
        n = self.Sf[sl] / self.magSf[sl][:, None]
        d = np.einsum('ij,ij->i', n, self.Cf[sl] - self.C[own])
        return dict(cells=own, Cf=self.Cf[sl], n=n, magSf=self.magSf[sl], delta=1.0 / np.abs(d))


# ----------------------------------------------------------------------- fields

_NCMPT = {'scalar': 1, 'vector': 3, 'symmTensor': 6, 'tensor': 9}


def _parse_value(txt, n_expected=None):
    """txt begins right after the keyword.  Handles 'uniform v', 'uniform (..)',
    'nonuniform List<type> N ( ... )' and 'nonuniform List<type> 0()'."""
    txt = txt.lstrip()
    if txt.startswith('uniform'):
        rest = txt[len('uniform'):].lstrip()
        if rest.startswith('('):
            v = _numbers(rest[:_matching(rest, 0)])
        else:
            v = np.array([float(re.match(_NUM, rest).group(0))])
        if n_expected is not None:
            v = np.tile(v, (n_expected, 1)) if v.size > 1 else np.full(n_expected, v[0])
        return v
    m = re.match(r'nonuniform\s+List<(\w+)>\s*(\d+)\s*', txt)
    if not m:
        raise ValueError('cannot parse field value: ' + txt[:60])
    nc, n = _NCMPT[m.group(1)], int(m.group(2))
    if n == 0:
        return np.zeros((0, nc)) if nc > 1 else np.zeros(0)
    i = m.end()
    v = _numbers(txt[i:_matching(txt, i)])
    return v.reshape(n, nc) if nc > 1 else v


def read_field(path, nCells=None, patch_sizes=None):
    """Returns (internal, {patch: {key: array}}) for keys value, valueFraction,
    refValue, refGradient, gradient (those present)."""
    b = _body(path)
    m = re.search(r'internalField\s+', b)
    internal = _parse_value(b[m.end():], nCells)
    out = {}
    m = re.search(r'boundaryField\s*\{', b)
    if m:
        end = _matching(b, m.end() - 1, '{', '}')
        blk = b[m.end():end - 1]
        i = 0
        for pm in re.finditer(r'([A-Za-z_"][\w.:|*()"-]*)\s*\{', blk):
            if pm.start() < i:
                continue
            j = _matching(blk, pm.end() - 1, '{', '}')
            name, sub = pm.group(1), blk[pm.end():j - 1]
            i = j
            ent = {}
            tm = re.search(r'\btype\s+([^;]+);', sub)
            ent['type'] = tm.group(1).strip() if tm else ''
            for key in ('value', 'valueFraction', 'refValue', 'refGradient', 'gradient'):
                km = re.search(r'(?<![\w])' + key + r'\s+(?=uniform|nonuniform)', sub)
                if km:
                    n = patch_sizes.get(name) if patch_sizes else None
                    ent[key] = _parse_value(sub[km.end():], n)
            out[name] = ent
    return internal, out


def header(cls, obj, loc=None):
    l = f'    location    "{loc}";\n' if loc else ''
    return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            f"    class       {cls};\n{l}    object      {obj};\n}}\n\n")


def scalar_list(v):
    v = np.atleast_1d(np.asarray(v, float))
    return f'nonuniform List<scalar> {len(v)}(' + ' '.join(f'{x:.17g}' for x in v) + ')'


def time_dirs(case, region=None):
    """Sorted (value, name) of the numeric time directories of a case."""
    out = []
    for d in os.listdir(case):
        try:
            v = float(d)
        except ValueError:
            continue
        if os.path.isdir(os.path.join(case, d)) and (region is None or os.path.isdir(os.path.join(case, d, region))):
            out.append((v, d))
    return sorted(out)
