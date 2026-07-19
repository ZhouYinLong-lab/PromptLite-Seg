# Third-Party Notices

The repository's Apache-2.0 license applies only to original PromptLite-Seg
code, documentation, and data-free metadata produced by the project. It does
not relicense third-party datasets, model code, model weights, or dependencies.

## PASCAL VOC 2012

- Project: <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/>
- Rights notice: <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/#rights>
- Local source mirror used by the preparation scripts:
  <https://huggingface.co/datasets/nateraw/pascal-voc-2012>

VOC images, semantic masks, target masks, and image-bearing visualizations are
not part of the clean public artifact. Users must obtain the dataset themselves
and comply with the original image and dataset terms. The committed protocol
manifests contain only row identifiers, class labels, target geometry, and
cryptographic hashes; they do not contain image or mask payloads. Use of the
Hugging Face mirror is a reproducibility convenience and grants no additional
rights.

Pinned parquet SHA-256 values:

- train: `7b3f275d2d634f6e2a8d1b82bee7f3b15491ade0d8fd60ae643b09478de2dbf2`
- validation: `6f3831b96a8c5705e7f5146c1e9c3e066f75020ab421b70ae7580a6929aa3722`

## Segment Anything

- Official source: <https://github.com/facebookresearch/segment-anything>
- Pinned commit: `dca509fe793f601edb92606367a655c15ac00fdf`
- Source archive SHA-256:
  `775a9fa2ea5441a7f532c77a9a193fec21764ef59e008bbdd595fee84e3aaab6`
- SAM ViT-B checkpoint SHA-256:
  `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`
- Upstream license: Apache License 2.0

The checkpoint is downloaded separately and is never committed to this
repository. Users remain responsible for complying with Meta's upstream terms.

## OpenCV

The GrabCut baseline uses `opencv-python-headless==4.10.0.84`. OpenCV is an
independent project distributed under its own license. See
<https://opencv.org/license/>.
