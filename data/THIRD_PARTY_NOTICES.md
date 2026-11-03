# Third-Party Notices

This directory includes inference code from the projects below. Original
copyright headers and license texts remain in the source trees. The repository
root license does not replace a component's own terms.

| Component | Source revision | License |
| --- | --- | --- |
| [CaRe-Ego](https://github.com/yuggiehk/CaRe-Ego) | `80c7402795bcbecdc16344ae025e34d0bb290fdc` | [Apache-2.0](third_party/CaRe-Ego/LICENSE.txt) |
| [SAM 2](https://github.com/facebookresearch/sam2) | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | [Apache-2.0](third_party/sam2/LICENSE), [cctorch notice](third_party/sam2/LICENSE_cctorch) |
| [ProPainter](https://github.com/sczhou/ProPainter) | `e870e79321c31b733e2031af5aa2fb1fe3ac7eec` | [NTU S-Lab License 1.0](third_party/ProPainter/LICENSE), non-commercial use |
| [DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser) | `8e6f279ac7531e27ad1849c6f8dab5372a8597e7` with modifications | [Apache-2.0](vendor/diffueraser/LICENSE) |

CaRe-Ego is by Yuejiao Su, Yi Wang, and Lap-Pui Chau. SAM 2 is from Meta's
Segment Anything project. ProPainter is by Shangchen Zhou, Chongyi Li,
Kelvin C. K. Chan, and Chen Change Loy. DiffuEraser is by Xiaowen Li,
Haolan Xue, Peiran Ren, and Liefeng Bo.

The included source trees are inference subsets. Training tools, benchmark
assets, and standalone upstream setup instructions are omitted. Component
README files direct users to this integration's setup and weight guide.

The DiffuEraser copy integrates temporal-window batching, batched VAE decoding,
and HDF5-backed frame access. Its `diffueraser/diffueraser.py` and
`diffueraser/pipeline_diffueraser.py` contain these integration changes.
Comments in `libs/brushnet_CA.py` are translated into English.
ProPainter's unused `RAFT/corr.py` CUDA-autograd wrapper is omitted; its default
correlation implementation is unchanged.
Other source files retain their upstream notices and any integration changes
already present in the included source versions.

The ProPainter prior is used by both the DiffuEraser and low-memory profiles.
Its license restricts use to non-commercial purposes. Pretrained weights and
optional models retain their source repositories' terms; see
[weights/README.md](weights/README.md).
