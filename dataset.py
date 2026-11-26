import rawpy
import numpy as np
import torch
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import os
import copy

def pack_raw(raw, clip_neg=True):
    im = raw.raw_image_visible.astype(np.float32) - 512.0
    if clip_neg:
        im = np.maximum(im, 0.0)
    im = im / (16383.0 - 512.0)
    im = np.expand_dims(im, axis=2)
    H, W, _ = im.shape

    out = np.concatenate((
        im[0:H:2, 0:W:2, :],  # R
        im[0:H:2, 1:W:2, :],  # G1
        im[1:H:2, 1:W:2, :],  # B
        im[1:H:2, 0:W:2, :]   # G2
    ), axis=2)
    return out


def get_exposure_time(filepath):
    fn = os.path.basename(filepath)
    return float(fn.split('_')[-1].lower().replace('s.arw', ''))


def raw4_to_srgb(raw4, wb=None):
    r = raw4[0]
    g = 0.5 * (raw4[1] + raw4[3])
    b = raw4[2]

    if wb is not None and len(wb) >= 3 and wb[1] > 0:
        r = r * (wb[0] / wb[1])
        b = b * (wb[2] / wb[1])

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.power(rgb, 1.0 / 2.2)
    return (rgb * 255.0).astype(np.uint8)


class Dataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """
    def __init__(self, data_dir, list_file, mode='train', transform=None,
                 patch_size=256, max_images=0, ratio=0.985, size_window=(5, 5)):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.patch_size = patch_size
        self.ratio = ratio
        self.size_window = size_window
        self.pairs = []

        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    short_rel = parts[0][2:] if parts[0].startswith('./') else parts[0].lstrip('/')
                    long_rel = parts[1][2:] if parts[1].startswith('./') else parts[1].lstrip('/')
                    short_path = os.path.join(data_dir, short_rel)
                    long_path = os.path.join(data_dir, long_rel)
                    if os.path.exists(short_path) and os.path.exists(long_path):
                        self.pairs.append((short_path, long_path))

        if max_images > 0 and max_images < len(self.pairs):
            indices = np.linspace(0, len(self.pairs) - 1, max_images, dtype=int)
            self.pairs = [self.pairs[i] for i in indices]

        if len(self.pairs) == 0:
            raise RuntimeError(f"Nessun file .ARW trovato! Controlla data_dir={data_dir} e list_file={list_file}")
            
    def __getitem__(self, index):
        short_path, long_path = self.pairs[index]
        ratio = min(get_exposure_time(long_path) / get_exposure_time(short_path), 300.0)

        with rawpy.imread(short_path) as raw_in:
            in_raw = np.clip(pack_raw(raw_in, clip_neg=False) * ratio, -1.0, 1.0)

        if self.mode == 'train':
            H, W, _ = in_raw.shape
            ps = self.patch_size
            yy = np.random.randint(0, H - ps + 1)
            xx = np.random.randint(0, W - ps + 1)
            patch = in_raw[yy:yy + ps, xx:xx + ps, :]

            label = patch.copy()
            inp, mask = self.generate_mask(patch)
            data = {'label': label, 'input': inp, 'mask': mask}
        else:
            with rawpy.imread(long_path) as raw_gt:
                gt_raw = np.clip(pack_raw(raw_gt, clip_neg=True), 0.0, 1.0)
                wb = np.array(raw_gt.camera_whitebalance[:3], dtype=np.float32)

            data = {
                'label': gt_raw,
                'input': in_raw,
                'wb': wb,
                'name': os.path.basename(short_path)
            }

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.pairs)

    def generate_mask(self, input):
        ratio = self.ratio
        size_window = self.size_window
        H, W, C = input.shape
        num_sample = int(H * W * (1 - ratio))

        mask = np.ones((H, W, C), dtype=np.float32)
        output = input.copy()

        idy_msk = np.random.randint(0, H, num_sample)
        idx_msk = np.random.randint(0, W, num_sample)

        half_y = size_window[0] // 2
        half_x = size_window[1] // 2
        idy_neigh = np.random.randint(-half_y, half_y + 1, num_sample)
        idx_neigh = np.random.randint(-half_x, half_x + 1, num_sample)

        same = (idy_neigh == 0) & (idx_neigh == 0)
        while np.any(same):
            idy_neigh[same] = np.random.randint(-half_y, half_y + 1, np.sum(same))
            idx_neigh[same] = np.random.randint(-half_x, half_x + 1, np.sum(same))
            same = (idy_neigh == 0) & (idx_neigh == 0)

        idy_msk_neigh = (idy_msk + idy_neigh) % H
        idx_msk_neigh = (idx_msk + idx_neigh) % W

        output[idy_msk, idx_msk, :] = input[idy_msk_neigh, idx_msk_neigh, :]
        mask[idy_msk, idx_msk, :] = 0.0

        return output, mask

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        out = {
            'input': torch.from_numpy(np.ascontiguousarray(data['input'].transpose((2, 0, 1)).astype(np.float32))),
            'label': torch.from_numpy(np.ascontiguousarray(data['label'].transpose((2, 0, 1)).astype(np.float32)))
        }
        if 'mask' in data:
            out['mask'] = torch.from_numpy(np.ascontiguousarray(data['mask'].transpose((2, 0, 1)).astype(np.float32)))
        if 'wb' in data:
            out['wb'] = torch.from_numpy(data['wb'].astype(np.float32))
        if 'name' in data:
            out['name'] = data['name']
        return out

class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input, label, mask = data['input'], data['label'], data['mask']

        input = (input - self.mean) / self.std
        label = (label - self.mean) / self.std

        data = {'input': input, 'label': label, 'mask': mask}
        return data


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        input, label, mask = data['input'], data['label'], data['mask']

        if np.random.rand() > 0.5:
            input = np.fliplr(input)
            label = np.fliplr(label)
            mask = np.fliplr(mask)

        if np.random.rand() > 0.5:
            input = np.flipud(input)
            label = np.flipud(label)
            mask = np.flipud(mask)

        return {'input': input, 'label': label, 'mask': mask}


class Rescale(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label = data['input'], data['label']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    input = transform.resize(input, (new_h, new_w))
    label = transform.resize(label, (new_h, new_w))

    return {'input': input, 'label': label}


class RandomCrop(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
    id_x = np.arange(left, left + new_w, 1).astype(np.int32)

    # input = input[top: top + new_h, left: left + new_w]
    # label = label[top: top + new_h, left: left + new_w]

    input = input[id_y, id_x]
    label = label[id_y, id_x]
    mask = mask[id_y, id_x]

    return {'input': input, 'label': label, 'mask': mask}


class UnifromSample(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, stride):
    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
      self.stride = (stride, stride)
    else:
      assert len(stride) == 2
      self.stride = stride

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]
    stride_h, stride_w = self.stride
    new_h = h//stride_h
    new_w = w//stride_w

    top = np.random.randint(0, stride_h + (h - new_h * stride_h))
    left = np.random.randint(0, stride_w + (w - new_w * stride_w))

    id_h = np.arange(top, h, stride_h)[:, np.newaxis]
    id_w = np.arange(left, w, stride_w)

    input = input[id_h, id_w]
    label = label[id_h, id_w]
    mask = mask[id_h, id_w]

    return {'input': input, 'label': label, 'mask': mask}


class ZeroPad(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    l = (new_w - w)//2
    r = (new_w - w) - l

    u = (new_h - h)//2
    b = (new_h - h) - u

    input = np.pad(input, pad_width=((u, b), (l, r), (0, 0)))
    label = np.pad(label, pad_width=((u, b), (l, r), (0, 0)))
    mask = np.pad(mask, pad_width=((u, b), (l, r), (0, 0)))

    return {'input': input, 'label': label, 'mask': mask}

class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}

class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = self.std * data + self.mean
        return data
