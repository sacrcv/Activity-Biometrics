"""ABNet: Activity-Biometrics -- Person Identification from Daily Activities.

Implementation of https://arxiv.org/abs/2403.17360 (CVPR 2024).

    Azad & Rawat. Activity-Biometrics: Person Identification from Daily
    Activities. CVPR 2024, pages 287-296.

The pipeline has three stages, run in order:

    1. ``tools/extract_silhouettes.py`` -- Mask2Former turns RGB clips into
       aligned binary silhouettes.
    2. ``train_teacher.py``            -- GaitGL learns identities from those
       silhouettes, becoming the bias-less teacher ``T``.
    3. ``train.py``                    -- ABNet trains against the frozen
       teacher, with biometrics distortion and joint activity learning.
"""

__version__ = "0.1.0"
