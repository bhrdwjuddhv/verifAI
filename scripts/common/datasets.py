"""One dataset index, two transforms.

`torchvision.datasets.ImageFolder` carries its transform, so the usual way to give a train and
a validation split different augmentation is to build the folder twice. That walks and stats
every file twice: on a 279k-file tree of symlinks (Kaggle) it is ~10 minutes of pure waiting
before the first epoch, for an index that is already in memory.

TransformSubset holds ONE index and applies its own transform, so the second walk disappears.
Nothing about the split, the labels or the augmentation changes — `selfcheck` asserts the
tensors are identical to the two-index version, because "equivalent" is the whole claim.
"""

from PIL import Image
from torch.utils.data import Dataset


class TransformSubset(Dataset):
    """A view over some indices of an ImageFolder, with its own transform.

    Reads `base.samples` directly rather than delegating to `base[i]`, so the base may be
    built without a transform (and therefore only walked once).
    """

    def __init__(self, base, indices, transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, target = self.base.samples[self.indices[i]]
        # Same as torchvision's default pil_loader: open, convert, and let the handle close.
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def selfcheck():
    """Identical output to Subset(ImageFolder(root, tf)) — the only thing that matters here."""
    import os
    import tempfile

    import torch
    from torch.utils.data import Subset
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    with tempfile.TemporaryDirectory() as root:
        for label, shade in (("Fake", 40), ("Real", 200)):
            os.makedirs(os.path.join(root, label))
            for n in range(4):
                Image.new("RGB", (48, 40), (shade, shade + n * 5, 128)).save(
                    os.path.join(root, label, f"{label.lower()}_{n}.png")
                )

        indices = [0, 3, 5, 7]
        old = Subset(datasets.ImageFolder(root, tf), indices)   # two-index version
        new = TransformSubset(datasets.ImageFolder(root), indices, tf)

        assert len(old) == len(new) == len(indices)
        for i in range(len(indices)):
            old_x, old_y = old[i]
            new_x, new_y = new[i]
            assert old_y == new_y, f"label drifted at {i}: {old_y} != {new_y}"
            assert torch.equal(old_x, new_x), f"pixels drifted at {i}"

        # A transform of None yields the PIL image, and the index order is exactly as given.
        raw = TransformSubset(datasets.ImageFolder(root), [5, 1], None)
        assert isinstance(raw[0][0], Image.Image)
        assert raw[0][1] == datasets.ImageFolder(root).samples[5][1]
        assert raw[1][1] == datasets.ImageFolder(root).samples[1][1]

    print("datasets selfcheck passed")


if __name__ == "__main__":
    selfcheck()
