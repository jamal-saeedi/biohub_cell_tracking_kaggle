"""The two shipped model sets and the config that runs them.

Both variants run the same pipeline settings (the `IsotropicConfig` defaults);
they differ only in their six checkpoints and six member-own edge re-scorers.

* ``fin``  -- the team's selected final submission (default).
* ``best`` -- the same recipe on each model's best-validation checkpoint.

The files live on Kaggle Models as one variation per variant
(``<KAGGLE_MODEL>/<handle>/<version>``) and are laid out as::

    <model root>/<handle>/<version>/isotropic_lineage/<run>/best.pt
    <model root>/<handle>/<version>/rescorers/<name>.npz

Every file is verified against its sha256 before it is loaded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from biohub_tracking.isotropic.config import IsotropicConfig

__all__ = ["DEFAULT_VARIANT", "KAGGLE_MODEL", "VARIANTS", "Variant", "shipped_config",
           "variant_dir", "verify_variant"]

KAGGLE_MODEL = "jamalsaeedi/biohub-cell-tracking/pyTorch"
DEFAULT_VARIANT = "fin"


@dataclass(frozen=True)
class Variant:
    handle: str
    version: int
    runs: tuple[str, ...]
    """Checkpoint runs, primary first."""
    rescorers: tuple[str, ...]
    """Member-own re-scorers, one per run, in the same order."""
    sha256: dict[str, str]
    """Relative path -> sha256 of every file of the variant."""

    @property
    def kaggle_handle(self) -> str:
        return f"{KAGGLE_MODEL}/{self.handle}/{self.version}"


VARIANTS: dict[str, Variant] = {
    "fin": Variant(
        handle="mres-fin-444322",
        version=1,
        runs=(
            "R3-ft24-noisy-lc-s1",
            "B3-v11-ft32-noisy-lc-s31-last",
            "D2-v11-ft32-noisy-lc-s33-last",
            "EX-ms-lc-s5-last",
            "FX-ms-lc-s6-last",
            "GX-ms-lc-s7-e23",
        ),
        rescorers=tuple(f"MRES-FIN-444322-m{k}" for k in range(6)),
        sha256={
            "isotropic_lineage/R3-ft24-noisy-lc-s1/best.pt": "6037a3b59f7903af4fd639d5c188ceff7cb5fbe1371893c63ab1fd466e11ec88",
            "isotropic_lineage/B3-v11-ft32-noisy-lc-s31-last/best.pt": "3d30a6a108b2d38a725c708d455c802e499e891e75de5c0579b5292cd48e8de9",
            "isotropic_lineage/D2-v11-ft32-noisy-lc-s33-last/best.pt": "33cffffae7d7b3a4a4b4722d4099b98636c40d122c0968a33ffe9e891cb533a2",
            "isotropic_lineage/EX-ms-lc-s5-last/best.pt": "73f815f902d6fffca74662ec5cb0ebbce70c2f7b07548fa6ec2b45f91603551c",
            "isotropic_lineage/FX-ms-lc-s6-last/best.pt": "8fc7e84bf122772f7adb4fd557c831055e209aadbe4955e9144caab72d325cd3",
            "isotropic_lineage/GX-ms-lc-s7-e23/best.pt": "d70da7c8174d391fcbf1228195f1d609789a75326ba977be41a896c5fb5f366f",
            "rescorers/MRES-FIN-444322-m0.npz": "f44d60e5977c0e5f1deb16d4e5eac40fe4612bd20ddc5a1d742999be55851f42",
            "rescorers/MRES-FIN-444322-m1.npz": "4147bac4d993faea12b1e7b1b3d51f0b612977883d4f077b0aebddb10a278b07",
            "rescorers/MRES-FIN-444322-m2.npz": "8b3a4add0567a8c1afd35c7e1c3dd6589ccf6e9a356726dfe54755f0d9d2c967",
            "rescorers/MRES-FIN-444322-m3.npz": "6661cd45c053c576a77a296dbcd222c503c767bf09ebfd149d3349650eaf9365",
            "rescorers/MRES-FIN-444322-m4.npz": "de5e3b10d8595f5d4fc1b37c0d713da197bbf0f5249a88654a8e3f9de95fd149",
            "rescorers/MRES-FIN-444322-m5.npz": "e5871fd2bc76cc7f57283f51d0633851c3d9a399436140816814a8af0135ec7e",
        },
    ),
    "best": Variant(
        handle="mres-best-444322",
        version=1,
        runs=(
            "R3-ft24-noisy-lc-s1",
            "B3-v11-ft32-noisy-lc-s31-best",
            "D2-v11-ft32-noisy-lc-s33-best",
            "EX-ms-lc-s5-best",
            "FX-ms-lc-s6-best",
            "GX-ms-lc-s7-best",
        ),
        rescorers=tuple(f"MRES-BEST-444322-m{k}" for k in range(6)),
        sha256={
            "isotropic_lineage/R3-ft24-noisy-lc-s1/best.pt": "6037a3b59f7903af4fd639d5c188ceff7cb5fbe1371893c63ab1fd466e11ec88",
            "isotropic_lineage/B3-v11-ft32-noisy-lc-s31-best/best.pt": "ae1f1cd4bbcb1721fbd1ab3c882a467a2bafac0aabd36a6da3106c5274bac070",
            "isotropic_lineage/D2-v11-ft32-noisy-lc-s33-best/best.pt": "51a8b2cf91ec876fdb5262190c27fccf03f967f06947edbb867c0b800a16de5d",
            "isotropic_lineage/EX-ms-lc-s5-best/best.pt": "712cf38e02287106c235f5a4e090da3b02e3896ce448c40a7a32e0537b977d11",
            "isotropic_lineage/FX-ms-lc-s6-best/best.pt": "5c617e666010ceee50e512d90190e390c1bb9f130edc048ff8a775f67f9b0d5a",
            "isotropic_lineage/GX-ms-lc-s7-best/best.pt": "d70da7c8174d391fcbf1228195f1d609789a75326ba977be41a896c5fb5f366f",
            "rescorers/MRES-BEST-444322-m0.npz": "de03b262957d89ca83824dfe299111f086d90161b28bc1cc1466be512e23e388",
            "rescorers/MRES-BEST-444322-m1.npz": "392a94bfc2fa1948507472bb6dbf3d51184d86e47c6386771f7196b31314a4af",
            "rescorers/MRES-BEST-444322-m2.npz": "e14a61bb6af2d560cb435db480773ad1e1d9c2d469b61e397025ace501fdfca0",
            "rescorers/MRES-BEST-444322-m3.npz": "a2820a3413c92c7d0a89d33ec6d5a72ba58d88ce7cfaf6e14c9d369a5fb3bc6e",
            "rescorers/MRES-BEST-444322-m4.npz": "4aafb01d6e04158d8cddcd267ec61823ab154eb26bbe565b09094cc810bfd777",
            "rescorers/MRES-BEST-444322-m5.npz": "11707dca61a7ba569ce28daef6cc98db837a37de897c4c054668bf6c95ab5125",
        },
    ),
}


def variant_dir(variant: str, model_roots: Path | str | list[Path | str]) -> Path:
    """The directory holding `variant`'s files: the first of `<root>/<handle>/<version>`,
    `<root>/<handle>` or `<root>` itself that contains `isotropic_lineage/`."""
    spec = VARIANTS[variant]
    roots = model_roots if isinstance(model_roots, list) else [model_roots]
    tried = []
    for root in map(Path, roots):
        for candidate in (root / spec.handle / str(spec.version), root / spec.handle, root):
            if (candidate / "isotropic_lineage").is_dir():
                return candidate
            tried.append(candidate)
    raise FileNotFoundError(
        f"no files for variant {variant!r} ({spec.kaggle_handle}); looked at:\n  "
        + "\n  ".join(map(str, tried))
        + "\nDownload them with `python tools/download_models.py`.")


def verify_variant(variant: str, directory: Path) -> None:
    """Raise unless every file of `variant` under `directory` has its pinned sha256."""
    for relative, expected in VARIANTS[variant].sha256.items():
        path = Path(directory) / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing model file {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"{path}: sha256 {digest} != pinned {expected}")


def shipped_config(variant: str = DEFAULT_VARIANT, model_roots: Path | str | list = (),
                   *, verify: bool = True) -> IsotropicConfig:
    """`IsotropicConfig()` with `variant`'s checkpoints and re-scorers attached."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    spec = VARIANTS[variant]
    directory = variant_dir(variant, list(model_roots) if isinstance(model_roots, (list, tuple))
                            else model_roots)
    if verify:
        verify_variant(variant, directory)
    checkpoints = [directory / "isotropic_lineage" / run / "best.pt" for run in spec.runs]
    return replace(
        IsotropicConfig(),
        checkpoint=checkpoints[0],
        ensemble_checkpoints=tuple(checkpoints[1:]),
        ensemble_rescorers=tuple(directory / "rescorers" / f"{name}.npz" for name in spec.rescorers),
    )
