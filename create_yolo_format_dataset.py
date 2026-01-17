import cv2
import os
import glob
import json
# import shutil
from pprint import pprint

if __name__ == "__main__":
    # Cấu hình đường dẫn
    # root_path = "../dataset/football_train"
    root_path = "../dataset/football_test"
    output_path = "../football_yolo_dataset"
    # is_train = True
    is_train = False

    # for subdir in os.listdir(root_path) :
    #     for file_ in os.listdir(os.path.join(root_path, subdir)) :
    #         print(file_)
    video_parts = list(glob.iglob("{}/*/*.mp4".format(root_path)))
    anno_parts = list(glob.iglob("{}/*/*.json".format(root_path)))
    
    video_without_extension = [video_part.replace(".mp4", "") for video_part in video_parts]
    anno_without_extension = [anno_part.replace(".json", "") for anno_part in anno_parts]

    parts = list(set(video_without_extension) & set(anno_without_extension))

    mode = "train" if is_train else "val"

    # if not os.path.isdir(output_path) and is_train :
    #     os.makedirs(output_path)
    #     os.makedirs(os.path.join(output_path, "images", mode))
    #     os.makedirs(os.path.join(output_path, "labels", mode))
    # elif not is_train :
    #     os.makedirs(os.path.join(output_path, "images", mode))
    #     os.makedirs(os.path.join(output_path, "labels", mode)) 


    # Tạo thư mục an toàn với exist_ok=True
    os.makedirs(os.path.join(output_path, "images", mode), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", mode), exist_ok=True)

    for idx, part in enumerate(parts) :
        video_path = "{}.mp4".format(part)
        json_path = "{}.json".format(part)

        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        with open(json_path, "r") as json_file :
            json_data = json.load(json_file)

            if num_frames != len(json_data["images"]):
                print(f"\n[Warning] Frame mismatch in {part}. Skipped.")
                video.release()
                continue
            

            # cac frame co width, height giong nhau nen lay frame dau tien
            width =json_data["images"][0]["width"]
            height = json_data["images"][0]["height"]

            # filter only player and ball objects
            all_objects = [{"image_id": obj["image_id"], "bbox": obj["bbox"], "category_id": obj["category_id"]} for obj in json_data["annotations"] if obj["category_id"] in [3, 4]]
  
            # group objects by frame
            objects_by_frame = {}
            for obj in all_objects:
                frame_idx = obj["image_id"] - 1
                if frame_idx not in objects_by_frame:
                    objects_by_frame[frame_idx] = []
                objects_by_frame[frame_idx].append(obj)

            frame_counter = 0
            while video.isOpened():
                print(idx, frame_counter)
                flag, frame = video.read()
                if not flag :
                    break
                
                # current_objects = [obj for obj in all_objects if obj["image_id"] - 1 == frame_counter]

                # Get objects for the current frame
                current_objects = objects_by_frame.get(frame_counter, [])


                # pprint(current_objects)
                # for obj in current_objects :
                    # x, y, w, h = obj["bbox"]
                    # cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 0), 2)

                    # xmin, ymin, w, h = obj["bbox"]
                    # cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmin+w), int(ymin+h)), (255, 0, 0), 2)


                cv2.imwrite(os.path.join(output_path, "images", mode, "{}_{}.jpg".format(idx, frame_counter)), frame)

                with open(os.path.join(output_path, "labels", mode, "{}_{}.txt".format(idx, frame_counter)), "w") as f :
                    for obj in current_objects :
                        xmin, ymin, w_box, h_box = obj["bbox"]

                        # Convert bbox to YOLO format
                        x_center = (xmin + w_box / 2) / width
                        y_center = (ymin + h_box / 2) / height
                        w = w_box / width
                        h = h_box / height

                        # Class 0: player, Class 1: ball
                        category = 0 if obj["category_id"] == 4 else 1

                        # Convert to YOLO format: class_id x_center y_center width height
                        f.write("{} {:6f} {:6f} {:6f} {:6f}\n".format(category, x_center, y_center, w, h))


                frame_counter += 1
        video.release()
    
    print("Done!")