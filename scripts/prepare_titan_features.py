"""
Convert the released TCGA TITAN slide features into the per-patient .pt files
that EMMS reads from data/titan_embeddings/.

The features come from the TITAN foundation model (Mahmood Lab):
    https://github.com/mahmoodlab/TITAN
Access is gated: request it on https://huggingface.co/MahmoodLab/TITAN and run
`hf auth login` (older versions: `huggingface-cli login`) before running this script.

Usage:
    python scripts/prepare_titan_features.py                 # download + convert
    python scripts/prepare_titan_features.py --inspect       # print pkl structure, convert nothing

The release is slide-level and EMMS is patient-level, so patients with several
slides are deduplicated on the first occurrence.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch


def load_pickle(path=None):
    """Return the unpickled TCGA TITAN feature object."""
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("MahmoodLab/TITAN", filename="TCGA_TITAN_features.pkl")
        print(f"Downloaded to {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def describe(obj, name='data', depth=0):
    """Print enough of the object's shape to see how it is keyed."""
    pad = '  ' * depth
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{pad}{name}: dict, {len(keys)} keys, e.g. {keys[:3]}")
        if keys and depth < 2:
            describe(obj[keys[0]], f"{name}[{keys[0]!r}]", depth + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__}, len {len(obj)}")
        if obj and depth < 2:
            describe(obj[0], f"{name}[0]", depth + 1)
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray {obj.shape} {obj.dtype}")
    elif isinstance(obj, torch.Tensor):
        print(f"{pad}{name}: tensor {tuple(obj.shape)} {obj.dtype}")
    else:
        print(f"{pad}{name}: {type(obj).__name__} = {str(obj)[:80]}")


def to_slide_dict(data):
    """Normalize the pickle into {slide_or_case_id: 1-D float array}.

    Handles the two shapes this file has shipped in: a plain
    {id: embedding} dict, and a dict with parallel 'filenames'/'embeddings'
    (or 'slide_ids'/'features') arrays.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict at the top level, got {type(data).__name__}")

    id_keys = ('filenames', 'slide_ids', 'slide_id', 'ids', 'index')
    feat_keys = ('embeddings', 'features', 'feats', 'X')
    id_key = next((k for k in id_keys if k in data), None)
    feat_key = next((k for k in feat_keys if k in data), None)

    if id_key and feat_key:
        ids, feats = data[id_key], data[feat_key]
        if len(ids) != len(feats):
            raise ValueError(f"{id_key} has {len(ids)} entries but {feat_key} has {len(feats)}")
        pairs = zip(ids, feats)
    else:
        pairs = data.items()

    out = {}
    for sid, emb in pairs:
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        emb = np.asarray(emb, dtype='float64').squeeze()
        if emb.ndim != 1:
            raise ValueError(f"{sid}: expected a 1-D embedding, got shape {emb.shape}")
        out[str(sid)] = emb
    return out


def patient_id(slide_id):
    """TCGA-A3-3308-01Z-00-DX1.<uuid> / TCGA-A3-3308-01 -> TCGA-A3-3308."""
    return '-'.join(str(slide_id).split('-')[:3])


def main(args):
    data = load_pickle(args.pickle)

    if args.inspect:
        describe(data)
        return

    slides = to_slide_dict(data)
    dims = {len(v) for v in slides.values()}
    print(f"Loaded {len(slides)} slide embeddings, dim(s) {sorted(dims)}")
    if len(dims) > 1:
        sys.exit(f"ERROR: inconsistent embedding dims {sorted(dims)}")

    # One slide per patient: keep the first occurrence, same as
    # drop_duplicates(subset='submitter_id', keep='first') in data_loader.py
    by_patient = {}
    n_dropped = 0
    for sid, emb in slides.items():
        pid = patient_id(sid)
        if pid in by_patient:
            n_dropped += 1
        else:
            by_patient[pid] = emb

    print(f"{len(by_patient)} patients ({n_dropped} extra slides dropped)")

    os.makedirs(args.out_dir, exist_ok=True)
    for pid, emb in sorted(by_patient.items()):
        torch.save(torch.tensor(emb, dtype=torch.float32), os.path.join(args.out_dir, f"{pid}.pt"))

    print(f"Wrote {len(by_patient)} files to {args.out_dir}")


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='Build data/titan_embeddings/ from the TITAN release')
    parser.add_argument('--pickle', type=str, default=None,
                        help='Local TCGA_TITAN_features.pkl (default: download from HuggingFace)')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(project_root, 'data', 'titan_embeddings'),
                        help='Where to write the per-patient .pt files')
    parser.add_argument('--inspect', action='store_true',
                        help='Print the pickle structure and exit')
    main(parser.parse_args())
