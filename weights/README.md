This directory is reserved for optional large pretrained weights.

- `vgg19-dcbb9e9d.pth` is used only when perceptual loss requires a local VGG checkpoint.
- The full file is about 548 MB, so it is not suitable for a normal GitHub repository commit.
- To download it locally, run:

```bash
bash scripts/download_vgg19.sh
```
