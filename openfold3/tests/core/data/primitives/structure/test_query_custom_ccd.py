"""Tests for building query structures from a user-supplied CCD (ccd_file_path).

Each test serves a renamed copy of a component from Biotite's internal CCD through a
custom CCD file, and checks that it builds the same way as the original component.
"""

import random

import biotite.structure as struc
import numpy as np
import pytest
import torch
from biotite.structure import info
from biotite.structure.io import pdbx

from openfold3.core.data.framework.single_datasets.inference import InferenceDataset
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_of3,
)
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_structure_of3,
)
from openfold3.core.data.primitives.structure.component import CustomCCDWithFallback
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Query

CUSTOM_LIGAND = "Z9ATP"  # copy of ATP
CUSTOM_RESIDUE = "Z9MSE"  # copy of MSE (selenomethionine)


def _renamed_ccd_block(source_code: str, new_code: str) -> pdbx.CIFBlock:
    block = pdbx.CIFBlock()
    for category_name in ["chem_comp", "chem_comp_atom", "chem_comp_bond"]:
        category = info.get_from_ccd(category_name, source_code)
        assert category is not None
        columns = {key: category[key].as_array().astype(str) for key in category}
        for key in ("id", "comp_id"):
            if key in columns:
                columns[key][:] = new_code
        block[category_name] = pdbx.CIFCategory(columns)
    return block


@pytest.fixture(scope="module")
def custom_ccd(tmp_path_factory) -> pdbx.CIFFile:
    all_residues = set(info.all_residues())
    assert CUSTOM_LIGAND not in all_residues
    assert CUSTOM_RESIDUE not in all_residues

    ccd = pdbx.CIFFile(
        {
            CUSTOM_LIGAND: _renamed_ccd_block("ATP", CUSTOM_LIGAND),
            CUSTOM_RESIDUE: _renamed_ccd_block("MSE", CUSTOM_RESIDUE),
        }
    )
    # Round-trip through a file, as ccd_file_path does
    path = tmp_path_factory.mktemp("ccd") / "custom_ccd.cif"
    ccd.write(path)
    return pdbx.CIFFile.read(path)


def _build(chains: list[dict], ccd: pdbx.CIFFile | None):
    query = Query.model_validate({"chains": chains})
    return structure_with_ref_mols_from_query(query, rng=random.Random(0), ccd=ccd)


def _bond_set(atom_array: struc.AtomArray) -> set:
    assert atom_array.bonds is not None
    bonds = set()
    for i, j, bond_type in atom_array.bonds.as_array():
        a = (atom_array.res_id[i], atom_array.atom_name[i])
        b = (atom_array.res_id[j], atom_array.atom_name[j])
        bonds.add((*sorted([a, b]), int(bond_type)))
    return bonds


def test_ligand_from_custom_ccd(custom_ccd):
    custom = _build(
        [{"molecule_type": "ligand", "chain_ids": ["A"], "ccd_codes": [CUSTOM_LIGAND]}],
        ccd=custom_ccd,
    )
    reference = _build(
        [{"molecule_type": "ligand", "chain_ids": ["A"], "ccd_codes": ["ATP"]}],
        ccd=None,
    )

    assert set(custom.atom_array.res_name) == {CUSTOM_LIGAND}
    np.testing.assert_array_equal(
        custom.atom_array.atom_name, reference.atom_array.atom_name
    )
    np.testing.assert_array_equal(
        custom.atom_array.element, reference.atom_array.element
    )
    assert _bond_set(custom.atom_array) == _bond_set(reference.atom_array)
    assert (
        custom.processed_reference_mols[0].mol.GetNumAtoms()
        == reference.processed_reference_mols[0].mol.GetNumAtoms()
    )


def test_non_canonical_residue_from_custom_ccd(custom_ccd):
    """Atoms, leaving atoms, intra-residue bonds and backbone links of a custom
    non-canonical residue match those of the equivalent built-in residue."""
    custom = _build(
        [
            {
                "molecule_type": "protein",
                "chain_ids": ["A"],
                "sequence": "GMG",
                "non_canonical_residues": {"2": CUSTOM_RESIDUE},
            }
        ],
        ccd=custom_ccd,
    )
    reference = _build(
        [
            {
                "molecule_type": "protein",
                "chain_ids": ["A"],
                "sequence": "GMG",
                "non_canonical_residues": {"2": "MSE"},
            }
        ],
        ccd=None,
    )

    assert list(np.unique(custom.atom_array.res_name)) == ["GLY", CUSTOM_RESIDUE]
    np.testing.assert_array_equal(
        custom.atom_array.atom_name, reference.atom_array.atom_name
    )
    assert _bond_set(custom.atom_array) == _bond_set(reference.atom_array)


def test_components_missing_from_custom_ccd_fall_back_to_biotite(custom_ccd):
    chains = [
        {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "GMG"},
        {"molecule_type": "ligand", "chain_ids": ["B"], "ccd_codes": ["ATP"]},
    ]
    with_ccd = _build(chains, ccd=custom_ccd)
    without_ccd = _build(chains, ccd=None)

    np.testing.assert_array_equal(
        with_ccd.atom_array.atom_name, without_ccd.atom_array.atom_name
    )
    assert _bond_set(with_ccd.atom_array) == _bond_set(without_ccd.atom_array)


def test_unknown_component_without_custom_ccd_raises():
    with pytest.raises(KeyError):
        _build(
            [
                {
                    "molecule_type": "ligand",
                    "chain_ids": ["A"],
                    "ccd_codes": [CUSTOM_LIGAND],
                }
            ],
            ccd=None,
        )


def test_inference_structure_features_match_builtin_components(custom_ccd):
    """End to end through InferenceDataset: features built from the custom CCD are
    the same as those built from the equivalent built-in components."""

    def structure_features(ligand: str, residue: str, ccd: pdbx.CIFFile | None):
        query = Query.model_validate(
            {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": "GMGAK",
                        "non_canonical_residues": {"2": residue},
                    },
                    {
                        "molecule_type": "ligand",
                        "chain_ids": ["B"],
                        "ccd_codes": [ligand],
                    },
                ]
            }
        )
        atom_array, ref_mols = InferenceDataset.get_structure_with_ref_mols(
            query, rng=random.Random(0), ccd=ccd
        )
        # Same featurization as InferenceDataset.create_structure_features
        target_features = featurize_structure_of3(
            atom_array=atom_array,
            n_tokens=int(atom_array.get_annotation("token_id").max()) + 1,
            is_gt=False,
            add_perm_features=False,
        )
        conformer_features = featurize_reference_conformers_of3(
            processed_ref_mol_list=ref_mols,
            add_ref_space_uid_to_perm=False,
            generator=torch.Generator().manual_seed(0),
        )
        return target_features | conformer_features

    custom = structure_features(CUSTOM_LIGAND, CUSTOM_RESIDUE, custom_ccd)
    reference = structure_features("ATP", "MSE", None)

    assert custom.keys() == reference.keys()
    # entity_id only differs because entities are numbered in sorted name order
    for name in custom.keys() - {"entity_id"}:
        torch.testing.assert_close(custom[name], reference[name], msg=name)


def test_custom_ccd_with_fallback(custom_ccd):
    ccd = CustomCCDWithFallback(custom_ccd)

    assert CUSTOM_LIGAND in ccd
    assert "ALA" in ccd
    assert "NOT_A_CCD_CODE" not in ccd
    assert ccd.get("NOT_A_CCD_CODE") is None

    custom_atoms = ccd[CUSTOM_LIGAND]["chem_comp_atom"]["atom_id"].as_array()
    builtin_atoms = info.get_from_ccd("chem_comp_atom", "ATP", "atom_id").as_array()
    np.testing.assert_array_equal(custom_atoms, builtin_atoms)

    alanine_atoms = ccd["ALA"]["chem_comp_atom"]["atom_id"].as_array()
    assert "CB" in alanine_atoms
