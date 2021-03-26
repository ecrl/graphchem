from re import compile
from .structures import Atom


def parse_smiles(smiles: str) -> list:

    # Identifiers for each character present in SMILES string
    link_re = compile(r'\d')
    organic_upper = ['B', 'C', 'N', 'O', 'P', 'S', 'F', 'B', 'I']
    organic_upper_2nd_letter = ['l', 'r']
    organic_lower = ['b', 'c', 'n', 'o', 'p', 's']
    bond_symb = {
        '.': 0.5,
        '-': 1.0,
        '=': 2.0,
        '#': 3.0,
        '$': 4.0,
        ':': 1.5,
        '/': 1.0,
        '\\': 1.0
    }

    # Keeping track of branching, indicated by '(' or ')'
    branch_level = 0
    new_branch = False

    # Default bond type is single bond, or '-'
    bond_type = 1.0

    # "Standard" indicates anything not inside "[ ]" (e.g. not [Au])
    standard = True

    # MAIN LOOP: look at each character in SMILES string
    atoms = []
    atom_count = 0
    for char in smiles:

        # Not inside [ ]
        if standard:

            # New branch
            if char == '(':
                branch_level += 1
                new_branch = True

            # Closing current branch
            elif char == ')':
                branch_level -= 1
                new_branch = False

            # Higher order bond (> 1)
            elif char in bond_symb.keys():
                bond_type = bond_symb[char]

            # Ring structure, indicated by number; find atom that links
            elif link_re.match(char) is not None:
                link = int(char)
                atoms[-1]._link = link
                for atom in atoms[:-1]:
                    if atom._link == link:
                        atom.add_connection(atoms[-1], bond_type)
                        atoms[-1].add_connection(atom, bond_type)

            # Atom, either non-aromatic (uppercase) or aromatic (lowercase)
            elif char in organic_upper or char in organic_lower:

                # Create Atom: ID, symbol, current branch status
                new_atom = Atom(atom_count, char, branch_level)
                atom_count += 1

                # If lowercase, atom is aromatic; set boolean
                if char in organic_lower:
                    new_atom._is_aromatic = 1

                # Look backwards through existing atoms to find connections
                for atom in reversed(atoms):

                    # If prev. atom has same branch level
                    if atom._branch_level == branch_level:

                        # Not evaluating connections in a new branch
                        #   CREATE CONNECTION
                        if not new_branch:
                            atom.add_connection(new_atom, bond_type)
                            new_atom.add_connection(atom, bond_type)
                            break

                    # If previous atom's branch level is less than current
                    #   CREATE CONNECTION
                    elif atom._branch_level < branch_level:
                        atom.add_connection(new_atom, bond_type)
                        new_atom.add_connection(atom, bond_type)
                        new_branch = False
                        break

                # Housekeeping
                atoms.append(new_atom)
                bond_type = 1.0

            # Character is 2nd letter in organic atom symbol
            elif char in organic_upper_2nd_letter:

                if atoms[-1]._symb not in organic_upper:
                    raise ValueError(
                        'Check SMILES format/syntax: {}'.format(smiles)
                    )
                atoms[-1]._symb += char

            # New custom symbol/inorganic atom/charge/isotope
            elif char == '[':

                standard = False
                new_atom = Atom(atom_count, '', branch_level)
                atom_count += 1
                atoms.append(new_atom)

            # Unknown symbol
            else:
                raise ValueError(
                    'Unknown character in SMILES: {}, {}'.format(
                        char, smiles
                    ))

        # Previous character "[", new CuStOm SMILES element
        else:

            # Handles pos. or neg. charge
            if char == '+':
                atoms[-1]._charge += 1
            elif char == '-':
                atoms[-1]._charge -= 1
            # End of custom element
            elif char == ']':
                standard = True
            # Charge scalar
            elif link_re.match(char) is not None:
                atoms[-1]._charge *= int(char)
            # Some letter, add it to the atom symbol!
            else:
                atoms[-1]._symb += char

    # Done
    return atoms
