import cv2
import pandas as pd
from typing import Dict
import os
import threading
from queue import Queue
import glob
from sklearn.model_selection import train_test_split


def aspect_preserving_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # taken from: https://stackoverflow.com/a/44659589
    # also used in: https://github.com/PyImageSearch/imutils/blob/c12f15391fcc945d0d644b85194b8c044a392e0a/imutils/convenience.py#L65-L94

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image (note dim is (w,h) following the convention of opencv for resize: https://stackoverflow.com/a/27871067)
    # print(f"Interpolation mode: {inter}")
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def get_interp_mode(input_ss, desired_ss):
    if input_ss < desired_ss:
        # upsampling
        return cv2.INTER_CUBIC
    else:
        # downsampling
        return cv2.INTER_AREA


def shortest_side_resize(image, shortest_side_size):
    # cv2 uses numpy for manipulating images, so h, w is the shape order when calling .shape()
    # according to: https://stackoverflow.com/a/19098258
    assert len(image.shape) == 3
    (height, width) = image.shape[:2]
    if width < height:
        inter = get_interp_mode(input_ss=width, desired_ss=shortest_side_size)
        image = aspect_preserving_resize(
            image=image, width=shortest_side_size, inter=inter
        )
    else:
        inter = get_interp_mode(input_ss=height, desired_ss=shortest_side_size)
        image = aspect_preserving_resize(
            image=image, height=shortest_side_size, inter=inter
        )
    return image


def video_to_rgb(
    video_filename, out_dir, shortest_side_size, padding_length=12, ext="png"
):
    os.makedirs(out_dir, exist_ok=True)
    # Using format() to dynamically create the format string with padding_length
    # Using png by default to avoid lossy compression (can use jpg if you have space constraints or especially large dataset)
    file_template = "frame_{{0:0{padding}d}}.{ext}".format(
        padding=padding_length, ext=ext
    )
    reader = cv2.VideoCapture(video_filename)
    try:
        (
            success,
            frame,
        ) = reader.read()  # read first frame
        # print(f"{video_filename}, {success}, {out_dir}, shortest side resize: {shortest_side_size}")
        count = 0
        while success:
            out_filepath = os.path.join(out_dir, file_template.format(count))
            # sanity check
            assert frame.shape[0] == reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
            assert frame.shape[1] == reader.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame = shortest_side_resize(
                image=frame, shortest_side_size=shortest_side_size
            )
            # print(f"Writing frame to {out_filepath}")
            cv2.imwrite(out_filepath, frame)
            success, frame = reader.read()
            count += 1
    finally:
        # ensures reader.release() is called even if an exception is raised
        # we do not catch exceptions in the try block in order to continue to avoid ignoring dropped frames and interfering with the underlying frame rate (which may important for tasks like optic flow, etc), but rather stop the moment an exception is raised
        reader.release()
        # helps demarcate it was actually done for later checking
        open(os.path.join(out_dir, "done.txt"), mode="w").close()


def thread_job(queue, process_video_func, **kwargs):
    while not queue.empty():
        video_filename = queue.get()
        process_video_func(
            video_filename=video_filename,
            **kwargs,  # Pass all kwargs to the process_video_func
        )
        queue.task_done()


def run_queue(
    args,
    process_video_func,  # Function to process video files
    video_filenames=None,
    **kwargs,  # Accepts video_in_path, rgb_out_path, shortest_side_size, file_extension, and any additional kwargs for process_video_func
):
    file_extension = kwargs.get(
        "file_extension", ".mp4"
    )  # Default to ".mp4" if not specified

    if video_filenames is None:
        video_in_path = kwargs.get(
            "video_in_path", ""
        )  # Should handle case where video_in_path is not provided
        video_filenames = [
            file for file in os.listdir(video_in_path) if file.endswith(file_extension)
        ]
    else:
        assert all(
            file.endswith(file_extension) for file in video_filenames
        ), "All video filenames must end with the specified file extension."

    queue = Queue()
    for video_filename in video_filenames:
        queue.put(video_filename)

    for i in range(args.num_threads):
        worker = threading.Thread(
            target=thread_job,
            kwargs={
                "queue": queue,
                "process_video_func": process_video_func,
                **kwargs,  # Pass all kwargs to thread_job
            },
        )
        worker.start()

    queue.join()


def label_id_convert(label_to_id_file):
    label_to_id_df = pd.read_csv(label_to_id_file)
    label_to_id_dict: Dict[str, int] = dict()
    id_to_label_dict: Dict[int, str] = dict()
    for index, row in label_to_id_df.iterrows():
        label_to_id_dict[row["name"]] = int(row["id"])
        id_to_label_dict[int(row["id"])] = row["name"]
    return label_to_id_dict, id_to_label_dict


def plot_video(rows, cols, frame_list, plot_width, plot_height, title: str):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    fig = plt.figure(figsize=(plot_width, plot_height))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
        axes_pad=0.3,  # pad between axes in inch.
    )

    for index, (ax, im) in enumerate(zip(grid, frame_list)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(index)
    plt.suptitle(title)
    plt.show()


def assign_videos_to_sets(
    base_dir, train_frac=0.9, val_frac=0.05, test_frac=0.05, random_state=42, ext="mp4"
):
    """
    Assign videos to train, validation, and test sets without moving the files.

    Args:
    - base_dir: The base directory where the videos are located.
    - train_frac: Proportion of the dataset to include in the train split.
    - val_frac: Proportion of the dataset to include in the validation split.
    - test_frac: Proportion of the dataset to include in the test split.
    - random_state: Seed for the random number generator.
    - ext: The file extension of the videos (default is mp4).
    """
    if not os.path.isdir(base_dir):
        print(f"The specified base_dir '{base_dir}' does not exist.")
        return

    # Ensure the sum of the sizes is 1
    if train_frac + val_frac + test_frac != 1:
        print("The sum of train, validation, and test fractions must be 1.")
        return

    # Initialize lists for train, validation, and test video paths
    train_videos = []
    val_videos = []
    test_videos = []
    all_videos = []

    location_dirs = glob.glob(os.path.join(base_dir, "*_*"))

    for loc_dir in location_dirs:
        videos = glob.glob(os.path.join(loc_dir, f"*.{ext}"))
        all_videos.extend(
            videos
        )  # Extend all_videos list with current directory videos

        if len(videos) >= 3:
            train_val_videos, test_vids = train_test_split(
                videos, test_frac=test_frac, random_state=random_state
            )
            train_vids, val_vids = train_test_split(
                train_val_videos,
                test_frac=val_frac / (train_frac + val_frac),
                random_state=random_state,
            )
        elif len(videos) == 2:
            train_vids, test_vids = train_test_split(
                videos, test_frac=0.5, random_state=random_state
            )
            val_vids = []  # No videos for validation
        elif len(videos) == 1:
            train_vids = videos
            val_vids = []  # No videos for validation
            test_vids = []  # No videos for test
        else:  # No videos in the directory
            train_vids = []
            val_vids = []
            test_vids = []

        # Correctly extend lists with the videos from this iteration
        train_videos.extend(train_vids)
        val_videos.extend(val_vids)
        test_videos.extend(test_vids)

    # Ensure that the videos are unique and disjoint
    assert len(set(train_videos)) == len(train_videos)
    assert len(set(val_videos)) == len(val_videos)
    assert len(set(test_videos)) == len(test_videos)
    assert len(train_videos) + len(val_videos) + len(test_videos) == len(all_videos)
    assert set(train_videos).isdisjoint(set(val_videos))
    assert set(train_videos).isdisjoint(set(test_videos))
    assert set(val_videos).isdisjoint(set(test_videos))
    assert set(train_videos + val_videos + test_videos) == set(all_videos)

    return train_videos, val_videos, test_videos
