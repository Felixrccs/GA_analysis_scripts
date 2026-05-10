# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER PARAMETERS
# ============================================================

DB_PATH = Path("/Users/dimitry/Documents/Data/EZGA/9-superhero/database/data_base/end_4_4_1")
OUTPUT_DIR = Path("/Users/dimitry/Documents/Data/EZGA/9-superhero/database/data_base/end_4_4_1/analisis_4_4_1")

# Geometry thresholds
OH_CUTOFF = 1.20          # O-H covalent bond cutoff, Å
HB_HO_CUTOFF = 2.50       # H...O acceptor cutoff, Å
HB_OO_CUTOFF = 3.50       # O_donor...O_acceptor cutoff, Å
HB_ANGLE_MIN = 140.0      # O_donor-H...O_acceptor angle, degrees

# Optional: exclude atoms below this z from OH/Hbond analysis.
# Set to None if you want to analyze the full structure.
Z_EXCLUDE_BELOW = None
# Z_EXCLUDE_BELOW = 10.0

LOG_LEVEL = "INFO"


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger("oh_hbond_counter")


# ============================================================
# Database helpers
# ============================================================

def open_storage(db_path: Path):
    from sage_lib.IO.storage.hybrid import HybridStorage
    return HybridStorage(db_path, access="ro")


def read_database_energies(storage: Any, ids: List[Any]) -> np.ndarray:
    """
    Read energies from HybridStorage.

    Uses get_all_energies() if available, otherwise falls back to per-structure access.
    """
    if hasattr(storage, "get_all_energies"):
        energies = storage.get_all_energies()
        return np.asarray(energies, dtype=float)

    energies = []

    for oid in ids:
        if hasattr(storage, "get_energy"):
            e = storage.get_energy(oid)
        else:
            struct = storage.get(oid)
            e = getattr(struct, "energy", getattr(struct, "E", None))

            if e is None and hasattr(struct, "get_potential_energy"):
                try:
                    e = struct.get_potential_energy()
                except Exception:
                    e = None

        if e is None:
            raise ValueError(f"Could not read energy for database id {oid!r}")

        energies.append(float(e))

    return np.asarray(energies, dtype=float)


# ============================================================
# Geometry utilities
# ============================================================

def pbc_displacement(apm: Any, dr: np.ndarray) -> np.ndarray:
    """
    Apply PBC to a displacement vector if the AtomPositionManager supports it.
    """
    if hasattr(apm, "apply_pbc_to_displacement"):
        return np.asarray(apm.apply_pbc_to_displacement(dr), dtype=float)
    return np.asarray(dr, dtype=float)


def pbc_distance(apm: Any, r1: np.ndarray, r2: np.ndarray) -> float:
    """
    Distance between two positions using PBC if available.
    """
    dr = pbc_displacement(apm, np.asarray(r2) - np.asarray(r1))
    return float(np.linalg.norm(dr))


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle between vectors in degrees.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosang)))


def get_allowed_atom_mask(struct: Any) -> np.ndarray:
    """
    Returns True for atoms included in the OH/Hbond analysis.

    If Z_EXCLUDE_BELOW is None, all atoms are included.
    If Z_EXCLUDE_BELOW is a number, atoms with z < Z_EXCLUDE_BELOW are excluded.
    """
    apm = struct.AtomPositionManager
    pos = np.asarray(apm.atomPositions, dtype=float)

    mask = np.ones(len(pos), dtype=bool)

    if Z_EXCLUDE_BELOW is not None:
        mask &= pos[:, 2] >= float(Z_EXCLUDE_BELOW)

    return mask


# ============================================================
# OH and H-bond analysis
# ============================================================

def classify_oxygen_hydrogen_environment(struct: Any) -> Dict[str, Any]:
    """
    Count OH groups, H2O-like oxygens, bare oxygens, and O-H...O H-bonds.

    Definitions
    -----------
    OH group:
        oxygen with exactly one covalently bound H,
        where d(O,H) < OH_CUTOFF.

    H2O-like oxygen:
        oxygen with exactly two covalently bound H atoms.

    bare oxygen:
        oxygen with zero covalently bound H atoms.

    H-bond:
        O_d-H...O_a satisfying:
            d(O_d,H) < OH_CUTOFF
            d(H,O_a) < HB_HO_CUTOFF
            d(O_d,O_a) < HB_OO_CUTOFF
            angle(O_d-H...O_a) >= HB_ANGLE_MIN

    Important:
        Each H can form at most one H-bond.
        If several acceptors satisfy the criteria, the closest H...O acceptor is selected.
    """

    apm = struct.AtomPositionManager
    pos = np.asarray(apm.atomPositions, dtype=float)
    labels = list(apm.atomLabelsList)

    allowed_mask = get_allowed_atom_mask(struct)

    o_idx = [
        i for i, s in enumerate(labels)
        if s == "O" and allowed_mask[i]
    ]

    h_idx = [
        i for i, s in enumerate(labels)
        if s == "H" and allowed_mask[i]
    ]

    # --------------------------------------------------------
    # Covalent O-H connectivity
    # --------------------------------------------------------

    o_to_h: Dict[int, List[int]] = {}
    h_to_o: Dict[int, List[int]] = {}

    for io in o_idx:
        attached_h = []

        for ih in h_idx:
            d_oh = pbc_distance(apm, pos[io], pos[ih])

            if d_oh < OH_CUTOFF:
                attached_h.append(int(ih))

        o_to_h[int(io)] = attached_h

    for ih in h_idx:
        attached_o = []

        for io in o_idx:
            d_oh = pbc_distance(apm, pos[io], pos[ih])

            if d_oh < OH_CUTOFF:
                attached_o.append(int(io))

        h_to_o[int(ih)] = attached_o

    # --------------------------------------------------------
    # Oxygen classification
    # --------------------------------------------------------

    oh_oxygens = []
    h2o_like_oxygens = []
    bare_oxygens = []
    overcoordinated_oxygens = []

    for io in o_idx:
        n_h = len(o_to_h[int(io)])

        if n_h == 0:
            bare_oxygens.append(int(io))
        elif n_h == 1:
            oh_oxygens.append(int(io))
        elif n_h == 2:
            h2o_like_oxygens.append(int(io))
        else:
            overcoordinated_oxygens.append(int(io))

    # --------------------------------------------------------
    # H-bond counting
    # --------------------------------------------------------
    #
    # Each H is allowed to form at most one H-bond.
    # For each covalent donor pair O_d-H, choose the best acceptor O_a.
    # Best acceptor = shortest H...O_a distance among valid candidates.
    # --------------------------------------------------------

    hbonds: List[Tuple[int, int, int]] = []
    hbond_details: List[Dict[str, Any]] = []

    donor_oxygens = [
        int(io) for io in o_idx
        if len(o_to_h[int(io)]) >= 1
    ]

    used_hydrogens = set()

    for od in donor_oxygens:
        for h in o_to_h[od]:

            # Safety: one H can only be assigned once.
            if h in used_hydrogens:
                continue

            rD = pos[od]
            rH = pos[h]

            # For angle O_d-H...O_a:
            # vector H -> donor O
            v_HD = pbc_displacement(apm, rD - rH)

            candidates = []

            for oa in o_idx:
                oa = int(oa)

                if oa == od:
                    continue

                rA = pos[oa]

                d_HA = pbc_distance(apm, rH, rA)
                if d_HA > HB_HO_CUTOFF:
                    continue

                d_DA = pbc_distance(apm, rD, rA)
                if d_DA > HB_OO_CUTOFF:
                    continue

                # vector H -> acceptor O
                v_HA = pbc_displacement(apm, rA - rH)

                ang = angle_deg(v_HD, v_HA)

                if ang >= HB_ANGLE_MIN:
                    candidates.append(
                        {
                            "donor_O": int(od),
                            "H": int(h),
                            "acceptor_O": int(oa),
                            "d_OH": float(pbc_distance(apm, rD, rH)),
                            "d_HA": float(d_HA),
                            "d_DA": float(d_DA),
                            "angle": float(ang),
                        }
                    )

            if candidates:
                # Choose one H-bond per H.
                # Criterion: closest H...O acceptor.
                best = min(candidates, key=lambda x: x["d_HA"])

                hbonds.append(
                    (
                        int(best["donor_O"]),
                        int(best["H"]),
                        int(best["acceptor_O"]),
                    )
                )

                hbond_details.append(best)
                used_hydrogens.add(int(h))

    hbonds = sorted(set(hbonds))

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    h_indices_in_hbonds = [h for _, h, _ in hbonds]

    if len(h_indices_in_hbonds) != len(set(h_indices_in_hbonds)):
        raise RuntimeError(
            "Internal error: one hydrogen was assigned to more than one H-bond."
        )

    return {
        "n_O": int(len(o_idx)),
        "n_H": int(len(h_idx)),
        "n_OH": int(len(oh_oxygens)),
        "n_H2O_like": int(len(h2o_like_oxygens)),
        "n_bare_O": int(len(bare_oxygens)),
        "n_overcoordinated_O": int(len(overcoordinated_oxygens)),
        "n_Hbond": int(len(hbonds)),
        "OH_oxygen_indices": oh_oxygens,
        "H2O_like_oxygen_indices": h2o_like_oxygens,
        "bare_oxygen_indices": bare_oxygens,
        "overcoordinated_oxygen_indices": overcoordinated_oxygens,
        "Hbond_triplets": hbonds,
        "Hbond_details": hbond_details,
    }


# ============================================================
# Main database analysis
# ============================================================

def count_oh_and_hbonds_in_database(
    db_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Analyze all structures in the database and write:

        oh_hbond_counts.csv
        oh_hbond_details.json
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = open_storage(db_path)

    try:
        ids = list(storage.iter_ids())

        if not ids:
            raise RuntimeError(f"No structures found in database: {db_path}")

        logger.info("Database contains %d structures.", len(ids))

        energies = read_database_energies(storage, ids)

        if len(energies) != len(ids):
            raise ValueError(
                f"Number of energies ({len(energies)}) does not match number of structures ({len(ids)})."
            )

        rows = []
        details_by_structure = {}

        for i, oid in enumerate(ids):
            if (i + 1) % 500 == 0:
                logger.info("Processed %d / %d structures.", i + 1, len(ids))

            struct = storage.get(oid)

            info = classify_oxygen_hydrogen_environment(struct)

            rows.append(
                {
                    "structure_index": int(i),
                    "database_id": str(oid),
                    "energy": float(energies[i]),
                    "n_O": info["n_O"],
                    "n_H": info["n_H"],
                    "n_OH": info["n_OH"],
                    "n_H2O_like": info["n_H2O_like"],
                    "n_bare_O": info["n_bare_O"],
                    "n_overcoordinated_O": info["n_overcoordinated_O"],
                    "n_Hbond": info["n_Hbond"],
                }
            )

            details_by_structure[str(oid)] = {
                "structure_index": int(i),
                "database_id": str(oid),
                "energy": float(energies[i]),
                "OH_oxygen_indices": info["OH_oxygen_indices"],
                "H2O_like_oxygen_indices": info["H2O_like_oxygen_indices"],
                "bare_oxygen_indices": info["bare_oxygen_indices"],
                "overcoordinated_oxygen_indices": info["overcoordinated_oxygen_indices"],
                "Hbond_triplets": [
                    [int(od), int(h), int(oa)]
                    for od, h, oa in info["Hbond_triplets"]
                ],
                "Hbond_details": info["Hbond_details"],
            }

        df = pd.DataFrame(rows)

        out_csv = output_dir / "oh_hbond_counts.csv"
        out_json = output_dir / "oh_hbond_details.json"

        df.to_csv(out_csv, index=False)

        with open(out_json, "w") as f:
            json.dump(details_by_structure, f, indent=2)

        logger.info("Saved count table: %s", out_csv)
        logger.info("Saved detailed H-bond data: %s", out_json)

        return df

    finally:
        storage.close()


# ============================================================
# Plotting
# ============================================================

def make_oh_hbond_plots(df: pd.DataFrame, output_dir: Path):
    """
    Make publication-style plots for OH and H-bond counts.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Plot 1: structure-index ordering
    # --------------------------------------------------------

    df_idx = df.sort_values("structure_index").reset_index(drop=True)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 1.2]},
    )

    axes[0].plot(
        df_idx["structure_index"],
        df_idx["n_OH"],
        linewidth=1.2,
        label="OH",
    )
    axes[0].plot(
        df_idx["structure_index"],
        df_idx["n_H2O_like"],
        linewidth=1.2,
        label="H2O-like O",
    )
    axes[0].set_ylabel("Count")
    axes[0].set_title("OH and H2O-like oxygen counts")
    axes[0].legend()

    axes[1].plot(
        df_idx["structure_index"],
        df_idx["n_Hbond"],
        linewidth=1.2,
        label="O-H···O H-bonds",
    )
    axes[1].set_ylabel("Count")
    axes[1].set_title("Hydrogen-bond count")
    axes[1].legend()

    axes[2].plot(
        df_idx["structure_index"],
        df_idx["energy"],
        linewidth=1.0,
        label="Energy",
    )
    axes[2].set_xlabel("Structure index")
    axes[2].set_ylabel("Energy")
    axes[2].set_title("Energy")
    axes[2].legend()

    plt.tight_layout()

    out1_png = output_dir / "oh_hbond_counts_vs_structure_index.png"
    out1_pdf = output_dir / "oh_hbond_counts_vs_structure_index.pdf"

    plt.savefig(out1_png, dpi=300, bbox_inches="tight")
    plt.savefig(out1_pdf, bbox_inches="tight")
    plt.show()

    # --------------------------------------------------------
    # Plot 2: energy-sorted ordering
    # --------------------------------------------------------

    df_e = df.sort_values("energy").reset_index(drop=True)
    x = np.arange(len(df_e))

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.2, 1.2]},
    )

    dE = df_e["energy"].to_numpy(dtype=float) - float(df_e["energy"].min())

    axes[0].plot(
        x,
        dE,
        linewidth=1.1,
    )
    axes[0].set_ylabel(r"$\Delta E$")
    axes[0].set_title("Structures sorted by energy")

    axes[1].plot(
        x,
        df_e["n_OH"],
        linewidth=1.2,
        label="OH",
    )
    axes[1].plot(
        x,
        df_e["n_H2O_like"],
        linewidth=1.2,
        label="H2O-like O",
    )
    axes[1].set_ylabel("Count")
    axes[1].set_title("OH / H2O-like oxygen counts")
    axes[1].legend()

    axes[2].plot(
        x,
        df_e["n_Hbond"],
        linewidth=1.2,
        label="O-H···O H-bonds",
    )
    axes[2].set_ylabel("Count")
    axes[2].set_title("Hydrogen-bond count")
    axes[2].legend()

    axes[3].scatter(
        df_e["n_OH"],
        df_e["n_Hbond"],
        s=12,
        alpha=0.6,
    )
    axes[3].set_xlabel("OH count")
    axes[3].set_ylabel("H-bond count")
    axes[3].set_title("Correlation between OH groups and H-bonds")

    plt.tight_layout()

    out2_png = output_dir / "oh_hbond_counts_energy_sorted.png"
    out2_pdf = output_dir / "oh_hbond_counts_energy_sorted.pdf"

    plt.savefig(out2_png, dpi=300, bbox_inches="tight")
    plt.savefig(out2_pdf, bbox_inches="tight")
    plt.show()

    logger.info("Saved plots:")
    logger.info("  %s", out1_png)
    logger.info("  %s", out1_pdf)
    logger.info("  %s", out2_png)
    logger.info("  %s", out2_pdf)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    df_counts = count_oh_and_hbonds_in_database(
        db_path=DB_PATH,
        output_dir=OUTPUT_DIR,
    )

    make_oh_hbond_plots(
        df=df_counts,
        output_dir=OUTPUT_DIR,
    )

    print("\nDone.")
    print(f"Count table:      {OUTPUT_DIR / 'oh_hbond_counts.csv'}")