
class Bond(object):

    def __init__(self, atom1, atom2, bondtype):

        self.atom1 = atom1
        self.atom2 = atom2
        self.bondtype = bondtype

    @property
    def repr_numeric(self):

        return (self.atom1._id, self.atom2._id, self.bondtype)


class Atom(object):
    def __init__(self, id, atom_symb, branch_level):
        self._id = id
        self._symb = atom_symb
        self._branch_level = branch_level
        self._bonds = []
        self._link = None
        self._charge = 0
        self._is_aromatic = 0

    def add_connection(self, atom, bond):

        self._bonds.append(Bond(self, atom, bond))

    @property
    def connectivity(self):

        return [
            [b.repr_numeric[0] for b in self._bonds],
            [b.repr_numeric[1] for b in self._bonds]
        ]
