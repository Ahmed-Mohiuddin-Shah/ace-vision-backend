import json
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import cv2

from helpers import save_frame_pil, update_progress

bw_method = 0.5


def produce_heatmaps(
    playerTop, playerBottom, progress_store, bw_method=0.5, task_id=None
):

    total_frames = len(playerTop) + len(playerBottom)
    processed = 0

    os.makedirs("results", exist_ok=True)

    image = cv2.imread("generate_heatmap/minimap.png")

    no_of_rows = image.shape[0]
    no_of_cols = image.shape[1]

    dataT = np.zeros((no_of_rows, no_of_cols))
    dataB = np.zeros((no_of_rows, no_of_cols))

    # # import playerTop and playerBottom and mark those coordinates as 1 in data
    # with open("playerTop.json", "r") as f:
    #     playerTop = json.load(f)

    # with open("playerBottom.json", "r") as f:
    #     playerBottom = json.load(f)

    for i in range(len(playerTop)):
        processed += 1
        if task_id:
            percent = 70 + int(20 * (processed / total_frames))
            update_progress(
                progress_store, task_id, percent, f"Heatmap {processed}/{total_frames}"
            )
        x = int(playerTop[i][0])
        y = int(playerTop[i][1])
        dataT[y][x] = 1

    for i in range(len(playerBottom)):
        processed += 1
        if task_id:
            percent = 70 + int(20 * (processed / total_frames))
            update_progress(
                progress_store, task_id, percent, f"Heatmap {processed}/{total_frames}"
            )

        x = int(playerBottom[i][0])
        y = int(playerBottom[i][1])
        dataB[y][x] = 1

    # for top player
    # Extract the indices of the non-zero elements
    yT, xT = np.nonzero(dataT)
    pointsT = np.vstack([xT, yT])

    # Create a kernel density estimate
    kT = gaussian_kde(pointsT, bw_method=bw_method)

    # apply simple meshgrid to get 2D arrays
    xiT, yiT = np.meshgrid(
        np.linspace(0, no_of_cols, no_of_cols), np.linspace(0, no_of_rows, no_of_rows)
    )

    # # Evaluate the density at each point in the grid
    ziT = kT(np.vstack([xiT.flatten(), yiT.flatten()]))

    # for bottom player
    # Extract the indices of the non-zero elements
    yB, xB = np.nonzero(dataB)
    pointsB = np.vstack([xB, yB])

    # Create a kernel density estimate
    kB = gaussian_kde(pointsB, bw_method=bw_method)

    # apply simple meshgrid to get 2D arrays
    xiB, yiB = np.meshgrid(
        np.linspace(0, no_of_cols, no_of_cols), np.linspace(0, no_of_rows, no_of_rows)
    )

    # # Evaluate the density at each point in the grid
    ziB = kB(np.vstack([xiB.flatten(), yiB.flatten()]))

    # for top player
    image2 = cv2.imread("generate_heatmap/minimap.png")

    ziT = ziT.reshape(xiT.shape)

    highest = np.max(ziT)

    def vectorized_heatmap(zi, image3, highest):

        norm = Normalize(vmin=0, vmax=highest)
        cmap = plt.get_cmap("jet")
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)
        image2 = scalar_map.to_rgba(zi, bytes=True)[:, :, :3]

        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

        return image2

    if task_id:
        percent = 94
        update_progress(
            progress_store, task_id, percent, f"Heatmap {processed}/{total_frames}"
        )
    print("Moving towards vectorization...  TOP")

    image3 = cv2.imread("generate_heatmap/minimap.png")

    image2 = vectorized_heatmap(ziT, image3, highest)

    for i in range(len(image3)):
        for j in range(len(image3[0])):
            if any(image3[i][j]) == 1:
                image2[i][j] = (255, 255, 255)

    save_frame_pil(image2, f"results/FINAL_heatmapT_{task_id}.png")

    # for bottom player
    image2 = cv2.imread("generate_heatmap/minimap.png")

    ziB = ziB.reshape(xiB.shape)

    highest = np.max(ziB)

    def vectorized_heatmap(zi, image3, highest):

        norm = Normalize(vmin=0, vmax=highest)
        cmap = plt.get_cmap("jet")
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)
        image2 = scalar_map.to_rgba(zi, bytes=True)[:, :, :3]

        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

        return image2

    if task_id:
        percent = 99
        update_progress(
            progress_store, task_id, percent, f"Heatmap {processed}/{total_frames}"
        )
    print("Moving towards vectorization...  TOP")

    image3 = cv2.imread("generate_heatmap/minimap.png")

    image2 = vectorized_heatmap(ziB, image3, highest)

    for i in range(len(image3)):
        for j in range(len(image3[0])):
            if any(image3[i][j]) == 1:
                image2[i][j] = (255, 255, 255)

    save_frame_pil(image2, f"results/FINAL_heatmapB_{task_id}.png")

    # combining both images
    ImageT = cv2.imread("results/FINAL_heatmapT.png")

    ImageB = cv2.imread("results/FINAL_heatmapB.png")

    # combining Top half of ImageT and bottom half of ImageB
    ImageT[no_of_rows // 2 :, :, :] = ImageB[no_of_rows // 2 :, :, :]

    save_frame_pil(ImageT, f"results/FINAL_heatmap_{task_id}.png")

    return ImageT
