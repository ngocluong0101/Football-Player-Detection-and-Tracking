import cv2
import os
import glob
import json
import shutil

if __name__ == "__main__":
    root_path = "../dataset/football_train"
    output_path = "./football_yolo_dataset"

    # for subdir in os.listdir(root_path) :
    #     for file_ in os.listdir(os.path.join(root_path, subdir)) :
    #         print(file_)
    video_parts = list(glob.iglob("{}/*/*.mp4".format(root_path)))
    anno_parts = list(glob.iglob("{}/*/*.json".format(root_path)))
    
    video_without_extension = [video_part.replace(".mp4", "") for video_part in video_parts]
    anno_without_extension = [anno_part.replace(".json", "") for anno_part in anno_parts]

    parts = list(set(video_without_extension) & set(anno_without_extension))

    if os.part.isdir(output_path) :
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "images"))
    os.makedirs(os.path.join(output_path, "labels"))

    for part in parts :
        video = cv2.VideoCapture("{}.mp4".format(part))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        with open("{}.json".format(part)) as json_file :
            json_data = json.load(json_file)
            if num_frames != len(json_data["images"]) :
                print("Frame number mismatch in {}".format(part))
                parts.remove(part)