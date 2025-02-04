import glob
import os
import re

root = "./outputs"


def get_ckpt_list(root):
    pt_files = glob.glob(f"{root}/**/**/*.pt", recursive=True)
    return pt_files


def get_root_path_list(path_list):
    res = [os.path.dirname(path) for path in path_list]
    return list(set(res))


if __name__ == "__main__":
    ckpt_list = get_ckpt_list(root)
    print(f"Total checkpoints found: {len(ckpt_list)}")

    ckpt_root_list = get_root_path_list(ckpt_list)
    ckpt_root_list.sort()
    for _root in ckpt_root_list:
        print(_root)
        ckpt_names = os.listdir(_root)
        ckpt_names.sort()
        ckpt_names = ckpt_names[:-2]
        for _ckpt_name in ckpt_names:
            ckpt_full_path = os.path.join(_root, _ckpt_name)
            print(
                ckpt_full_path,
                f"{os.path.getsize(ckpt_full_path) / 1024 / 1024 / 1024:.2f} GB",
            )
            assert os.path.exists(ckpt_full_path)
            try:
                os.remove(ckpt_full_path)
            except Exception as e:
                print(e)
        print("*" * 100)

    if os.path.exists("data/temp_fake/") and False:
        os.rmdir("data/temp_fake/")
        print("data/temp_fake/ removed")

    if False:
        for ckpt in ckpt_list:
            step_num = int(ckpt.split("/")[-1].replace("ckpt_", "").replace(".pt", ""))
            print(
                ckpt, step_num, f"{os.path.getsize(ckpt) / 1024 / 1024 / 1024:.2f} GB"
            )