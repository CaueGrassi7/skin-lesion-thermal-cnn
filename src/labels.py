"""Biopsy-confirmed label linkage for the TR_heatSkin dataset.

The on-disk labels are a *folder-name* heuristic (`benigna`→Healthy,
`lesao`→Sick). This module derives the **biopsy-confirmed** ground truth
instead, by linking each image folder to its row(s) in
`data/raw/Labels_Skin_Cancer.xlsx` (histopathology results). This is both more
rigorous (ground truth = biopsy, not folder naming) and removes label noise:
some `lesao` folders biopsied *benign*.

Task produced here: the **clean binary benign × cancer** framing —
- benign control folders (`benigna`/`benigno`) → benign (0);
- suspicious folders (`lesao*`) labelled by their lesion's `Tipo de câncer`:
  `Cancer` → 1, `Benign` → 0 (relabelled, flagged), `pre-cancer` → excluded.

Anything that cannot be resolved *unambiguously and consistently* from the
spreadsheet is **excluded** and reported for the advisor to adjudicate:
- patients with several `lesao` folders whose biopsy types differ
  (folder↔video cannot be matched from names alone);
- rows where `Tipo de câncer` contradicts the `Filtro Biópsia` diagnosis;
- image folders whose patient is absent from the spreadsheet.

Nothing here silently guesses a diagnosis — unresolved cases become
`label=None` (dropped from training) and appear in `write_report`.

Usage:
    from src.labels import build_label_map, write_report
    records = build_label_map(xlsx_path, data_root)
    label_map = {r.folder: r.label for r in records if r.label is not None}
    write_report(records, out_csv)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Histopathology keywords used only to CROSS-CHECK the `Tipo de câncer` column
# against the free-text `Filtro Biópsia` diagnosis, flagging disagreements — not
# to assign labels. Kept deliberately conservative (unambiguous terms only).
_MALIGNANT_TERMS = ["melanoma", "carcinoma", "cbc", "cec", "espinocelular", "basocelular"]
_BENIGN_TERMS = [
    "verruga", "nevo azul", "nevo melanocítico", "nevo melanocitico",
    "cisto", "dermatite", "ceratose seborreica", "queratose seborreica",
    "tricofoliculoma", "beninga", "benigno", "benigna",
]

LABEL_BENIGN = 0
LABEL_CANCER = 1


@dataclass
class FolderLabel:
    """Biopsy-linkage result for one image folder (one thermal sequence)."""

    folder: str
    folder_class: str  # "Healthy" | "Sick" (the on-disk heuristic)
    patient_id: str
    folder_type: str  # "benigna" | "lesao" | ...
    label: Optional[int]  # 0=benign, 1=cancer, None=excluded
    status: str  # see module docstring
    tipo: Optional[str] = None  # matched biopsy `Tipo de câncer`
    biopsias: list = field(default_factory=list)  # matched `Biópsia` codes
    filtros: list = field(default_factory=list)  # matched `Filtro Biópsia` texts
    note: str = ""


def _norm_pid(value) -> str:
    """Normalize a patient id (drop trailing '.0' from Excel floats, strip)."""
    return re.sub(r"\.0$", "", str(value)).strip()


def _folder_type(folder_name: str) -> str:
    """Classify a folder by its name: 'benigna' (control) or 'lesao' (suspect)."""
    lname = folder_name.lower()
    if "benign" in lname:
        return "benigna"
    if "lesao" in lname or "lesão" in lname:
        return "lesao"
    return "outro"


def _load_sheet(xlsx_path: Path) -> pd.DataFrame:
    """Load and normalize the biopsy spreadsheet to the columns we use."""
    df = pd.read_excel(xlsx_path)
    out = pd.DataFrame(
        {
            "pid": df["Id paciente"].map(_norm_pid),
            "video": df["ID vídeo"].astype(str).str.strip(),
            "local": df["Local"].astype(str).str.strip(),
            "biop": df["Biópsia"].astype(str).str.strip(),
            "tipo": df["Tipo de câncer"].astype(str).str.strip().str.lower(),
            "filt": df["Filtro Biópsia"].astype(str).str.strip(),
        }
    )
    # Drop fully-empty rows (trailing spreadsheet blanks)
    return out[out["pid"].ne("nan") & out["pid"].ne("")].reset_index(drop=True)


def _is_benign_control(biop: str) -> bool:
    """True for the spreadsheet's benign-control rows (`Biópsia == 'b'`)."""
    return biop.strip().lower() == "b"


def _tipo_to_label(tipo: str) -> tuple[Optional[int], str]:
    """Map a `Tipo de câncer` value to (label, status) for the clean binary task."""
    t = tipo.strip().lower()
    if t == "cancer":
        return LABEL_CANCER, "clean_cancer"
    if t == "benign":
        return LABEL_BENIGN, "relabeled_benign"  # suspect lesion, biopsy benign
    if t == "pre-cancer":
        return None, "excluded_precancer"
    return None, "excluded_unknown_tipo"


def _filtro_conflicts(label: int, filtros: list) -> bool:
    """True if the free-text diagnosis clearly contradicts the assigned label.

    A cancer label whose only diagnosis text reads benign (or vice versa) is a
    data-entry inconsistency that a human must resolve.
    """
    text = " ".join(str(f) for f in filtros).lower()
    if not text or text == "nan":
        return False
    has_malignant = any(w in text for w in _MALIGNANT_TERMS)
    has_benign = any(w in text for w in _BENIGN_TERMS)
    if label == LABEL_CANCER:
        return has_benign and not has_malignant
    if label == LABEL_BENIGN:
        return has_malignant and not has_benign
    return False


def build_label_map(xlsx_path: str | Path, data_root: str | Path) -> list[FolderLabel]:
    """Link every image folder under `data_root` to a biopsy-confirmed label.

    Args:
        xlsx_path: Path to `Labels_Skin_Cancer.xlsx`.
        data_root: Dataset root with `Healthy/` and `Sick/` subdirs of
            per-patient folders (raw or processed — only folder names are read).

    Returns:
        One `FolderLabel` per image folder. `label is None` marks folders
        excluded from the clean binary task (reason in `status`). Pass the
        non-None entries to `_collect_samples(..., label_map=...)`.
    """
    xlsx_path = Path(xlsx_path)
    data_root = Path(data_root)
    sheet = _load_sheet(xlsx_path)

    # Index spreadsheet rows by patient
    by_patient: dict[str, pd.DataFrame] = {pid: g for pid, g in sheet.groupby("pid")}

    # Count 'lesao' folders per patient up front (to detect the multi-lesion
    # ambiguity), scanning both class dirs once.
    lesao_folders_per_patient: dict[str, int] = {}
    for class_name in ("Healthy", "Sick"):
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            continue
        for folder in class_dir.iterdir():
            if not folder.is_dir() or _folder_type(folder.name) != "lesao":
                continue
            m = re.search(r"Paciente_?(\d+)", folder.name)
            fpid = m.group(1) if m else folder.name
            lesao_folders_per_patient[fpid] = lesao_folders_per_patient.get(fpid, 0) + 1

    records: list[FolderLabel] = []
    for class_name in ("Healthy", "Sick"):
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            continue
        for folder in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            name = folder.name
            m = re.search(r"Paciente_?(\d+)", name)
            pid = m.group(1) if m else name
            ftype = _folder_type(name)
            rows = by_patient.get(pid)

            if rows is None:
                records.append(FolderLabel(
                    name, class_name, pid, ftype, None, "missing_from_sheet",
                    note="patient absent from spreadsheet",
                ))
                continue

            if ftype == "benigna":
                # Benign control folder: benign by construction. Confirm a
                # benign-control row exists for transparency.
                benign_rows = rows[rows["biop"].map(_is_benign_control)]
                records.append(FolderLabel(
                    name, class_name, pid, ftype, LABEL_BENIGN, "clean_benign",
                    tipo="benign",
                    biopsias=benign_rows["biop"].tolist(),
                    filtros=benign_rows["filt"].tolist(),
                    note="" if len(benign_rows) else "no explicit 'b' row; benign by folder type",
                ))
                continue

            # Suspicious (`lesao`) folder → label from the suspicious rows.
            susp = rows[~rows["biop"].map(_is_benign_control)]
            n_lesao = lesao_folders_per_patient.get(pid, 1)
            tipos = sorted(set(susp["tipo"]))

            if len(susp) == 0:
                records.append(FolderLabel(
                    name, class_name, pid, ftype, None, "ambiguous_no_suspect_row",
                    note="lesao folder but no suspicious row in sheet",
                ))
            elif n_lesao > 1 and len(tipos) > 1:
                # Multiple lesao folders with differing biopsy types — cannot
                # map folder→video from names alone.
                records.append(FolderLabel(
                    name, class_name, pid, ftype, None, "ambiguous_multilesion",
                    biopsias=susp["biop"].tolist(), filtros=susp["filt"].tolist(),
                    note=f"{n_lesao} lesao folders, differing tipos {tipos}",
                ))
            else:
                # Either a single suspicious lesion, or several that agree on
                # tipo → the label is unambiguous even if folder↔video is not.
                tipo = tipos[0] if len(tipos) == 1 else susp["tipo"].iloc[0]
                label, status = _tipo_to_label(tipo)
                filtros = susp["filt"].tolist()
                if label is not None and _filtro_conflicts(label, filtros):
                    records.append(FolderLabel(
                        name, class_name, pid, ftype, None, "inconsistent",
                        tipo=tipo, biopsias=susp["biop"].tolist(), filtros=filtros,
                        note="Tipo de câncer contradicts Filtro Biópsia diagnosis",
                    ))
                else:
                    note = ""
                    if n_lesao > 1:
                        note = f"{n_lesao} lesao folders share tipo={tipo}"
                    records.append(FolderLabel(
                        name, class_name, pid, ftype, label, status,
                        tipo=tipo, biopsias=susp["biop"].tolist(), filtros=filtros,
                        note=note,
                    ))

    return records


def write_report(records: list[FolderLabel], out_path: str | Path) -> dict[str, int]:
    """Write a per-folder CSV report and return a status→count summary.

    The CSV lists every folder with its assigned label and status; the caller
    should surface the non-clean rows to the advisor. Returns the status tally
    (also useful for logging).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "folder": r.folder,
                "folder_class": r.folder_class,
                "patient_id": r.patient_id,
                "folder_type": r.folder_type,
                "label": r.label,
                "status": r.status,
                "tipo": r.tipo,
                "biopsias": "; ".join(map(str, r.biopsias)),
                "filtros": "; ".join(map(str, r.filtros)),
                "note": r.note,
            }
            for r in records
        ]
    )
    df.to_csv(out_path, index=False)
    return df["status"].value_counts().to_dict()
