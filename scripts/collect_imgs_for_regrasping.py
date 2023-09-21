import os
import sys
from pathlib import Path
from roboflow import Roboflow
from time import time, sleep

import bosdyn.client
import bosdyn.client.util
from PIL import Image
from bosdyn.client.image import ImageClient
from tqdm import tqdm

from conq.cameras_utils import get_color_img


def main(argv):
    sdk = bosdyn.client.create_standard_sdk('collect_imgs_for_regrasping')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
    project = rf.workspace("armlab").project("spot-vacuuming-demo")

    now = int(time())
    root = Path(f"data/{now}")
    root.mkdir(parents=True, exist_ok=True)
    print("Saving images to", root)

    filenames = []
    while True:
        image_client = robot.ensure_client(ImageClient.default_service_name)
        rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
        now = int(time())
        filename = root / f"{now}_rgb.png"
        Image.fromarray(rgb_np).save(filename)
        filenames.append(filename)

        k = input("Press enter to capture next image, or q to quit.")
        if k == 'q':
            break

    print("Uploading images to Roboflow...")
    for filename in tqdm(filenames):
        project.upload(
            image=str(filename),
            batch_name=str(root),
            num_retry_uploads=3
        )
    print("Done!")

    return True


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
