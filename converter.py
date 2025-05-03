#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
from pathlib import Path

def convert_recording(src: Path, dst: Path, is_sim: bool):
    """
    Read a MuJoCo-style recording.hdf5 and write an ACT-style episode_N.hdf5.
    """
    # Ensure output directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Open source and target files
    with h5py.File(src, 'r') as f_src, h5py.File(dst, 'w') as f_dst:
        # Root-level attr
        f_dst.attrs['sim'] = is_sim

        # Load arrays
        qpos  = f_src['observations/qpos'][()]
        qvel  = f_src['observations/qvel'][()]
        action= f_src['observations/ctrl'][()]

        # Create action dataset
        f_dst.create_dataset('action',
                             data=action,
                             compression='gzip')

        # Create observations group
        obs = f_dst.create_group('observations')
        obs.create_dataset('qpos', data=qpos,  compression='gzip')
        obs.create_dataset('qvel', data=qvel,  compression='gzip')

        # Copy camera images
        img_grp_src = f_src['observations/images']
        img_grp_dst = obs.create_group('images')
        for cam in img_grp_src:
            data = img_grp_src[cam][()]
            img_grp_dst.create_dataset(cam,
                                      data=data,
                                      compression='gzip',
                                      chunks=(1, *data.shape[1:]))

    print(f"Converted {src} → {dst}")


def main():
    p = argparse.ArgumentParser(
        description="Convert MuJoCo HDF5 recordings to ACT episodes"
    )
    p.add_argument('--input_dir',  '-i', required=True,
                   help="Folder containing subdirs with recording.hdf5")
    p.add_argument('--output_dir', '-o', required=True,
                   help="Folder to write episode_<n>.hdf5 files")
    p.add_argument('--sim',        action='store_true',
                   help="Mark episodes as simulation (sets sim=True)")
    args = p.parse_args()

    in_root  = Path(args.input_dir)
    out_root = Path(args.output_dir)

    # Iterate through all recording.hdf5 files
    for idx, src in enumerate(in_root.rglob('recording.hdf5')):
        dst = out_root / f"episode_{idx}.hdf5"
        convert_recording(src, dst, is_sim=args.sim)

if __name__ == "__main__":
    main()
