import os
import av
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def visualize_rows(parquet_path, video_path, start_row, end_row, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    # load parquet
    df = pd.read_parquet(parquet_path)

    print(f"\n{'='*60}")
    print(f"Parquet   : {parquet_path}")
    print(f"Video     : {video_path}")
    print(f"Total rows: {len(df)}")
    print(f"Showing rows {start_row} to {end_row-1}")
    print(f"{'='*60}")

    for i in range(start_row, min(end_row, len(df))):
        row = df.iloc[i]

        frame_index   = int(row["frame_index"])
        episode_index = int(row["episode_index"])
        timestamp     = float(row["timestamp"])
        state         = list(row["observation.state"])
        action        = list(row["action"])

        print(f"\n── row {i} ───────────────────────────────────────────")
        print(f"  frame_index   : {frame_index}")
        print(f"  episode_index : {episode_index}")
        print(f"  timestamp     : {timestamp:.4f}s")
        print(f"  state         : {[round(x, 4) for x in state]}")
        print(f"  action        : {[round(x, 4) for x in action]}")

        # decode the frame from mp4
        frame_img = None
        with av.open(video_path) as container:
            for frame in container.decode(video=0):
                if frame.time >= timestamp:
                    frame_img = Image.fromarray(
                        frame.to_ndarray(format="rgb24")
                    )
                    break

        if frame_img is None:
            print(f"  WARNING: could not find frame at timestamp {timestamp:.4f}s")
            continue

        # save png
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(frame_img)
        ax.set_title(
            f"row={i}  |  frame_index={frame_index}  |  episode={episode_index}\n"
            f"timestamp={timestamp:.4f}s\n"
            f"state={[round(x,3) for x in state]}\n"
            f"action={[round(x,3) for x in action]}",
            fontsize=7
        )
        ax.axis("off")
        plt.tight_layout()

        save_path = os.path.join(
            save_dir, f"row_{i:05d}_frame_{frame_index}_ep_{episode_index}.png"
        )
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {save_path}")

    print(f"\nDone. {end_row - start_row} images saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--parquet_path", type=str, required=True,
                        help="path to the parquet file")
    parser.add_argument("--video_path", type=str, required=True,
                        help="path to the corresponding mp4 file")
    parser.add_argument("--start_row", type=int, default=0,
                        help="which row to start from")
    parser.add_argument("--end_row", type=int, default=10,
                        help="which row to stop at (exclusive)")
    parser.add_argument("--save_dir", type=str, default="./viz_parquet")

    args = parser.parse_args()

    visualize_rows(
        parquet_path=args.parquet_path,
        video_path=args.video_path,
        start_row=args.start_row,
        end_row=args.end_row,
        save_dir=args.save_dir
    )