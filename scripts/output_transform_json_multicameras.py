import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def closest_point_2_lines(
    oa, da, ob, db
):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


if __name__ == "__main__":
    base_root = "data/showroom_undistortion-self"
    # base_root = '/mnt/diskb/liushiyong/Polycam/output12/colmap'
    # base_root = '/mnt/diskb/liushiyong/videos/IMG_0995'
    # base_root = '/mnt/diskb/liushiyong/videos/IMG_1032'
    # base_root = '/mnt/diskb/liushiyong/hw_cloud/1037_1039/colmap_undistort/'
    # TEXT_FOLDER = os.path.join(base_root, 'colmap/sparse/0')
    TEXT_FOLDER = os.path.join(base_root, "sparse/0")
    AABB_SCALE = 16
    SKIP_EARLY = 0
    keep_colmap_coords = False
    IMAGE_FOLDER = os.path.join(base_root, "images/")
    MASK_FOLDER = os.path.join(base_root, "masks/")
    EDGE_FOLDER = os.path.join(base_root, "edge/")
    DEPTH_FOLDER = os.path.join(base_root, "depth/")
    OUT_PATH = os.path.join(TEXT_FOLDER, "transforms.json")
    RENDER_OUPUT_PATH = os.path.join(TEXT_FOLDER, "render.json")
    cameras = {}
    with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
        angle_x = math.pi / 2
        for line in tqdm(f):
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            id = els[0]
            camera_model = els[1]
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[5])
            k1 = 0
            k2 = 0
            k3 = 0
            k4 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            is_fisheye = False
            distortion = []
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                fl_y = float(els[4])
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                distortion = [k1]
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
                distortion = [k1, k2]
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
                distortion = [k1, k2, p1, p2]
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                is_fisheye = True
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                distortion = [k1]
            elif els[1] == "RADIAL_FISHEYE":
                is_fisheye = True
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
                distortion = [k1, k2]
            elif els[1] == "OPENCV_FISHEYE":
                is_fisheye = True
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                k3 = float(els[10])
                k4 = float(els[11])
                distortion = [k1, k2, k3, k4]
            else:
                print("Unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi
            cameras[id] = {
                "camera_model": camera_model,
                "w": w,
                "h": h,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "distortion": distortion,
                "k1": k1,
                "k2": k2,
                "p1": p1,
                "p2": p2,
                "angle_x": angle_x,
                "angle_y": angle_y,
                "fovx": fovx,
                "fovy": fovy,
            }

    # print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2}")

    with open(os.path.join(TEXT_FOLDER, "images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            "camera_model": camera_model,
            "is_fisheye": is_fisheye,
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }

        # out_render = {
        #     "keyframes": [],
        #     "camera_type": "perspective",
        #     "render_height": 1080,
        #     "render_width": 1920,
        #     "camera_path": [],
        #     "fps": 30,
        #     "seconds": 8,
        #     "smoothness_value": 0,
        #     "is_cycle": False,
        #     "crop": None
        # }

        up = np.zeros(3)
        for line in tqdm(f):
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < SKIP_EARLY * 2:
                continue
            if i % 2 == 1:
                elems = line.split(
                    " "
                )  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                # name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(IMAGE_FOLDER)
                name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                # b = sharpness(name)
                # print(name, "sharpness=",b)
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = elems[8]

                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                if not keep_colmap_coords:
                    c2w[0:3, 2] *= -1  # flip the y and z axis
                    c2w[0:3, 1] *= -1
                    c2w = c2w[[1, 0, 2, 3], :]
                    c2w[2, :] *= -1  # flip whole world upside down

                    up += c2w[0:3, 1]

                frame = {
                    "file_path": str(f"{IMAGE_FOLDER}{'_'.join(elems[9:])}"),
                    # "sharpness": b,
                    "w": cameras[camera_id]["w"],
                    "h": cameras[camera_id]["h"],
                    "fl_x": cameras[camera_id]["fl_x"],
                    "fl_y": cameras[camera_id]["fl_y"],
                    "cx": cameras[camera_id]["cx"],
                    "cy": cameras[camera_id]["cy"],
                    "distortion": cameras[camera_id]["distortion"],
                    "k1": k1,
                    "k2": k2,
                    "p1": p1,
                    "p2": p2,
                    "transform_matrix": c2w,
                }
                out["frames"].append(frame)
    nframes = len(out["frames"])

    if keep_colmap_coords:
        flip_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(
                f["transform_matrix"], flip_mat
            )  # flip cameras (it just works)
    else:
        # don't keep colmap coords - reorient the scene to be easier to work with

        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(
                R, f["transform_matrix"]
            )  # rotate up to be the z axis

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3, :]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3, :]
                p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                if w > 0.00001:
                    totp += p * w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp)  # the cameras are looking at totp
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] -= totp

        avglen = 0.0
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

        with open(Path(base_root) / "dataparser_transforms.json", "w") as f:
            data = {"transform": R.tolist(), "scale": 4.0 / avglen}
            json.dump(data, f)

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
        if os.path.exists(MASK_FOLDER):
            f["mask_path"] = os.path.join(MASK_FOLDER, os.path.basename(f["file_path"]))
        if os.path.exists(EDGE_FOLDER):
            f["edge_path"] = os.path.join(EDGE_FOLDER, os.path.basename(f["file_path"]))
        if os.path.exists(DEPTH_FOLDER):
            f["depth_file_path"] = os.path.join(
                DEPTH_FOLDER, os.path.basename(f["file_path"])
            )
    print(nframes, "frames")
    print(f"writing {OUT_PATH}")

    out["frames"].sort(key=lambda x: x["file_path"])
    # for index, f in enumerate(out["frames"]):
    #     camera_path = np.array(f['transform_matrix']).reshape(-1).tolist()
    #     out_render['camera_path'].append({"camera_to_world": camera_path, "fov": 50, "aspect": 1.7152542372881356})
    #     out_render['keyframes'].append({"matrix": str(np.array(f['transform_matrix']).T.reshape(-1).tolist()), "fov": 50, "aspect": 1.7152542372881356, "properties": f"[[\"FOV\",50],[\"NAME\",\"Camera {index}\"],[\"TIME\",{index}]]"})

    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)

    # with open(RENDER_OUPUT_PATH, "w") as outfile:
    #     json.dump(out_render, outfile, indent=2)
    sys.exit(0)
