#import "@preview/kunskap:0.1.0": *

#set math.mat(delim: "[")
#set math.vec(delim: "[")

#set page(footer: context {
  set text(size: 8pt, fill: rgb("#888888"))
  grid(
    columns: (1fr, auto, 1fr),
    align(left)[#link("https://sytwu.github.io/SLAT-Studio/")[Project page: sytwu.github.io/SLAT-Studio]],
    align(center)[#counter(page).display()],
    [],
  )
})

#show: kunskap.with(
  title: "Analysis of TRELLIS for Image-to-3D Generation and SLAT-based Downstream Editing",
  author: "Group 2",
  header: "NYCU - Computer Vision 2026 Final Project",
  paper-size: "a4",
)

#v(-5em)

#figure(
  image("figures/trellis_teaser.png", width: 100%),
  caption: [*TRELLIS* [1] generates high-quality 3D assets from images or text, decoding a single latent into 3D Gaussians, radiance fields, or meshes.],
)

#v(-2em)

= *Problem Definition*

#v(-0.3em)
The diversity of 3D representations forces existing generative models to balance between *geometric accuracy* and *appearance fidelity*, which hinders a standardized paradigm like the one enjoyed by 2D generation. To address this, *TRELLIS* introduces a unified and versatile latent space, the *Structured LATent (SLAT)*, which is capable of high-quality 3D generation across diverse output representations, including 3D Gaussians, radiance fields, and meshes.

#v(-1em)

= *Motivation*

#v(-0.3em)
3D assets are essential for games, AR/VR, animation, and robotics simulation, yet traditional modeling demands professional skills and heavy manual effort. Image-to-3D methods like TRELLIS offer a convenient alternative. However, real-world inputs are often *not ideal*, transparency, metallic reflections, printed text, or thin parts are common in everyday objects but challenging to reconstruct, and the resulting loss of detail or incorrect geometry limits downstream use.

This motivates a detailed analysis of TRELLIS covering *both successful and failure cases* across object categories and materials, to understand its behavior and the causes of failure. Since TRELLIS represents objects with SLAT, we further probe its latent space (e.g., interpolation) and extend its downstream editing capabilities together with *RePaint* [2].

= *Related Work*

#v(-0.3em)
*3D generative models.*
Native 3D generative models learn priors directly within 3D or latent spaces. To overcome the high complexity and scaling limits of early voxels and Implicit Neural Representations (INRs), recent frameworks (e.g., TRELLIS [1], Hunyuan3D [5]) shift toward *3D latent generative models*. By compressing 3D data into a compact latent space via a 3D autoencoder, they train a diffusion or flow-matching transformer within this space to achieve efficient, high-quality 3D generation.

*3D creation with 2D generative models.*
Instead of directly training 3D generative models, some recent works leverage 2D diffusion priors to guide the 3D generation process. For instance, DreamFusion [3] optimizes 3D assets using an SDS loss, distilling knowledge from pre-trained image diffusion models.

*Rectified flow models.*
Because diffusion generation is inefficient, flow matching has recently become a novel generative paradigm that reduces the number of denoising steps. Among various flow-matching approaches, Rectified Flow [4] further speeds up inference without degraded performance. The original TRELLIS [1] paper utilizes Rectified Flow as base generative model.

#v(-1em)

= *Method*

#v(-0.3em)
TRELLIS encodes a 3D asset into *Structured LATents (SLAT)*, a set of active voxels, each carrying a local latent feature aggregated from multiview DINOv2 features. Generation runs in two stages. A *sparse-structure flow transformer* first produces the active-voxel layout from a text or image condition; a *structured-latent flow transformer* then fills in the per-voxel features. A sparse decoder finally converts SLAT into the desired output 3D Gaussians, radiance fields, or meshes. This shared latent is what enables our downstream editing tasks.

#figure(
  image("figures/trellis_pipeline.png", width: 100%),
  caption: [*TRELLIS pipeline* [1]. Top: a sparse VAE encodes 3D assets into SLAT and decodes them into multiple representations. Bottom: two flow transformers generate the sparse structure and the structured latents from a text or image condition.],
)

#pagebreak()

= *Experiments & Analysis*

== *1. Data Augmentation*

#v(-0.3em)
We evaluate the robustness of the TRELLIS image-to-3D pipeline under a range of input perturbations, including occlusion, Gaussian blur, rotation, background replacement, JPEG compression, downscaling, edge extraction, and horizontal flipping, to understand which corruptions it tolerates and which ones cause it to fail.

#figure(
  image("figures/comparison_typical_creature_dragon_expanded.png", width: 100%),
  caption: [*Input perturbations and their generated 3D results.* Each pair shows the perturbed input (left) and the corresponding TRELLIS reconstruction (right).],
)

== *2. Limitation*

#v(-0.3em)
=== *(a) Complicated texture (transparent, high-specular)*

#v(-0.3em)
TRELLIS struggles with objects whose appearance is dominated by *transparency* or *high specularity*. For such materials, single-image cues are ambiguous: the model cannot disambiguate the true surface from reflections or refractions, which leads to incorrect geometry and washed-out appearance.

#v(0.3em)
#figure(
  grid(
    columns: 2,
    column-gutter: 8pt,
    row-gutter: 6pt,
    image("figures/dice_row.png", width: 100%), image("figures/ice_cube_row.png", width: 100%),
    image("figures/speciman_row.png", width: 100%), image("figures/tea_pot_row.png", width: 100%),
  ),
  caption: [*Translucent and high-specular objects.* The transparent dice, ice cubes, glass specimen cube, and metallic kettle lose fine detail and produce distorted geometry.],
)

=== *(b) Object with text*

#v(-0.3em)
We found that TRELLIS *cannot faithfully generate objects that carry printed text*. The generated surface preserves the overall color and shape, but the text on the object is smeared into meaningless marks, likely because the latent representation does not allocate capacity to such high-frequency, semantically specific surface details.

#figure(
  grid(
    columns: 1,
    row-gutter: 6pt,
    image("figures/basketball_row.png", width: 100%),
    image("figures/book_row.png", width: 100%),
  ),
  caption: [*Text is not preserved.* The printed markings on the input ball and the text on the book degrade into illegible scribbles across the generated views.],
)

== *3. Downstream Tasks*

#v(-0.3em)
Building on the SLAT representation, we extend TRELLIS into a small suite of downstream tasks: three text-driven *editing* operations and one *latent-interpolation* task. Each takes one or more SLAT-encoded assets, optionally conditioned on a text prompt or a spatial editing region, and re-runs part of the generative pipeline.

=== *(a) Restyle*

#v(-0.3em)
- *Input:* 3D asset + text prompt.
- *Output:* the same geometry but with a different appearance.
- *Method:* 3D asset $arrow$ SLAT $arrow$ text-conditioned structured-latent re-generation $arrow$ result.

#v(-1.0em)
#figure(
  image("figures/restyle_chest.png", width: 60%),
  caption: [*Restyle.* A wooden chest (input) is restyled into solid gold while preserving geometry.],
)

=== *(b) Region Editing*

#v(-0.3em)
- *Input:* 3D asset + text prompt + an editing region (voxels).
- *Output:* region-specific modifications.
- *Method:* 3D asset $arrow$ SLAT $arrow$ repainting [2] at structured-latent generation $arrow$ result.

#v(-1.0em)
#figure(
  image("figures/region_edit.png", width: 80%),
  caption: [*Region editing.* Left: input. Middle: the selected region (green). Right: the region edited into "lava" while the rest of the asset is unchanged.],
)

=== *(c) Inpainting*

#v(-0.3em)
- *Input:* a carved (partially removed) 3D asset + text prompt + editing region (voxels).
- *Output:* the completed 3D asset.
- *Method:* carved 3D asset $arrow$ sparse structure $arrow$ repainting at *structure* generation $arrow$ repainting at *structured-latent* generation $arrow$ result.

Unlike region editing, which keeps the geometry fixed and repaints only the structured latents, inpainting must also recover the missing geometry, so RePaint is applied at *both* the sparse-structure and structured-latent stages.

#v(-1.0em)
#figure(
  image("figures/inpainting.png", width: 80%),
  caption: [*Inpainting.* Left: input. Middle: the carved region. Right: the completed asset.],
)

=== *(d) Latent Interpolation (Morph)*

#v(-0.3em)
- *Input:* two 3D assets + the number of intermediate points.
- *Output:* a gradual transition between the two assets.
- *Method:* two 3D assets $arrow$ SLATs $arrow$ interpolation on the union of their voxels with parameter $tau$ $arrow$ result.

#v(-1.5em)
#figure(
  image("figures/morph_chest.png", width: 100%),
  caption: [*Morph.* Interpolating two chests in SLAT space at $tau = 0.00, 0.25, 0.50, 0.75, 1.00$.],
)

#v(-1.0em)

= *Reference*

#v(-0.3em)
#text(size: 8pt)[
  #set enum(numbering: "[1]", spacing: 0.9em)
  + J. Xiang, Z. Lv, S. Xu, Y. Deng, R. Wang, B. Zhang, D. Chen, X. Tong, and J. Yang. Structured 3D Latents for Scalable and Versatile 3D Generation. In _Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)_, 2025.
  + A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool. RePaint: Inpainting using Denoising Diffusion Probabilistic Models. In _Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)_, 2022.
  + B. Poole, A. Jain, J. T. Barron, and B. Mildenhall. DreamFusion: Text-to-3D using 2D Diffusion. In _Int. Conf. on Learning Representations (ICLR)_, 2023.
  + X. Liu, C. Gong, and Q. Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In _Int. Conf. on Learning Representations (ICLR)_, 2023.
  + Tencent Hunyuan3D Team. Hunyuan3D 2.0: High-Resolution 3D Asset Generation. _arXiv preprint arXiv:2501.12202_, 2025.
]

#v(-1.0em)

= *Work Assignment Plan*

#v(-0.3em)
#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (center, center, left),
    table.header([*Member*], [*Student ID*], [*Contribution*]),
    [鄭名翔], [113950011], [Data augmentation experiments, Report refinement],
    [張媛婷], [113950020], [Limitation analysis (complicated texture & text), Report refinement],
    [司徒立中], [111550159], [SLAT downstream pipeline (Restyle / Region Editing), Report writing],
    [何義翔], [111550106], [Inpainting & Morph implementation, Asset collection],
    [曾紹幃], [111550040], [RePaint integration & Result visualization, Report refinement],
  ),
)
