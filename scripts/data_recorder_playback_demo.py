import pickle
from pathlib import Path

from conq.cameras_utils import image_to_opencv


def main():
    pkls_dir = Path("/media/big_narstie/datasets/conq_hose_manipulation_raw")
    print(pkls_dir.exists())
    pkls_paths = list(pkls_dir.rglob('*.pkl'))
    for pkl_path in pkls_paths:
        with pkl_path.open('rb') as f:
            data = pickle.load(f)

        for step in data[:10]:
            state = step['robot_state']
            snapshot = state.kinematic_state.transforms_snapshot
            timestamp = state.kinematic_state.acquisition_timestamp
            print(timestamp)
            # use snapshot to get the transforms

            for src, res in step['images'].items():
                rgb_np = image_to_opencv(res)
                print(src)
                # use src and rgb_np


if __name__ == "__main__":
    main()
