"""NITorchRegister support library.

Submodules are imported directly by callers (e.g.
``from NITorchRegisterLib.validation import compute_dice_scores``) rather than
re-exported here. This keeps the nitorch-free paths — Dice scoring
(``validation``) and the VTK grid-transform-node creation — importable without a
working nitorch install; only ``registration`` and ``grid_transform`` pull in
nitorch, and they do so lazily at their own call sites.
"""
