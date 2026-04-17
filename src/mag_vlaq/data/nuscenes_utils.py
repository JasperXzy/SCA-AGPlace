import os

import numpy as np
import torch
import torchvision.transforms as T
import utm
from PIL import Image

from mag_vlaq.models.layers.sparse_utils import sparse_quantize

trainselectlocationlist = ["singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown", "boston-seaport"]

testselectlocationlist = ["singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown", "boston-seaport"]


def get_location_from_sample_token(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    scene_token = sample["scene_token"]
    location = nusc.get("log", nusc.get("scene", scene_token)["log_token"])["location"]
    return location


def get_latloneastnorth_from_sample_token(nusc, sample_token, location):
    sample = nusc.get("sample", sample_token)
    ego_pose = nusc.get("ego_pose", sample["data"]["LIDAR_TOP"])
    ego_pose = ego_pose["translation"]
    ego_pose = np.array(ego_pose)
    if location == "boston-seaport":
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(42.336849169438615, -71.05785369873047)
        # clock-wise rotate
        degree = 1.5
        R = np.array(
            [
                [np.cos(np.pi / 180 * degree), -np.sin(np.pi / 180 * degree)],
                [np.sin(np.pi / 180 * degree), np.cos(np.pi / 180 * degree)],
            ]
        )
        ego_pose[0:2] = ego_pose[0:2] @ R
    elif location == "singapore-onenorth":
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(1.2882100868743724, 103.78475189208984)
    elif location == "singapore-hollandvillage":
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(1.2993652317780957, 103.78217697143555)
    elif location == "singapore-queenstown":
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(1.2782562240223188, 103.76741409301758)
    else:
        raise NotImplementedError
    ego_pose[0] = ego_pose[0] + eastingorg
    ego_pose[1] = ego_pose[1] + northingorg
    easting = ego_pose[0]
    northing = ego_pose[1]
    latlon = utm.to_latlon(easting, northing, zone_number, zone_letter)
    outputdict = {
        "lat": latlon[0],
        "lon": latlon[1],
        "east": easting,
        "north": northing,
        "zone_number": zone_number,
        "zone_letter": zone_letter,
    }
    return outputdict


def get_datapaths_from_sample_token(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    sensornames = [
        "LIDAR_TOP",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    datapaths = {}
    for sensorname in sensornames:
        datatoken = sample["data"][sensorname]
        data = nusc.get("sample_data", datatoken)
        dataname = data["filename"]
        datapath = os.path.join(nusc.dataroot, dataname)
        # assert os.path.exists(datapath), f'{datapath} does not exist'
        datapaths[sensorname] = datapath

    return datapaths


def load_sensordata_from_sampletoken(nusc, sample_token, args):
    sample = nusc.get("sample", sample_token)
    sensornames = [
        "LIDAR_TOP",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    sensordatas = {}
    for sensorname in sensornames:
        datatoken = sample["data"][sensorname]
        data = nusc.get("sample_data", datatoken)
        dataname = data["filename"]
        datapath = os.path.join(nusc.dataroot, dataname)
        if sensorname == "LIDAR_TOP":
            # load pc using .pcd.bin
            # pc = LidarPointCloud.from_file(datapath).points.T # [n,4]
            # pc = pc[:,:3] # [n,3]
            # load pc using .npy
            pcpath = datapath.replace(".pcd.bin", ".npy")
            pcpath = pcpath.split("/")
            pcpath[-2] += f"_voxel{1}"
            pcpath = "/".join(pcpath)
            pc = np.load(pcpath, allow_pickle=True)
            pc = sparse_quantize(coordinates=pc, quantization_size=args.quant_size)
            sensordatas["LIDAR_TOP"] = pc

            sensordatas["RANGE_DATA"] = torch.empty(0)

            # load bev
            # w = 200
            # max_thd = 100
            # bevdatapath = datapath.replace('.pcd.bin', '.png')
            # bevdatapath = bevdatapath.replace('samples/LIDAR_TOP', f'samples/BEV_DATA_w{w}_thd{max_thd}')
            # bev = Image.open(bevdatapath).convert('RGB')
            # tf = T.Compose([T.ToTensor(),
            #                 T.Resize(256, antialias=True), # [3,32,32]
            #                 ])
            # bev = tf(bev)
            # sensordatas['BEV_DATA'] = bev
            sensordatas["BEV_DATA"] = torch.empty(0)
        else:  # for cameras
            datapath = datapath.split("/")
            datapath[-2] += f"_size{256}"
            datapath = "/".join(datapath)
            image = Image.open(datapath)
            tf = T.Compose(
                [
                    T.Resize((args.q_resize, args.q_resize), antialias=True),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            image = tf(image)
            sensordatas[sensorname] = image

    # Stack
    qcam = []
    camnames = args.camnames
    for camname in camnames:
        if camname == "f":
            qcam.append(sensordatas["CAM_FRONT"])
        elif camname == "fl":
            qcam.append(sensordatas["CAM_FRONT_LEFT"])
        elif camname == "fr":
            qcam.append(sensordatas["CAM_FRONT_RIGHT"])
        elif camname == "b":
            qcam.append(sensordatas["CAM_BACK"])
        elif camname == "bl":
            qcam.append(sensordatas["CAM_BACK_LEFT"])
        elif camname == "br":
            qcam.append(sensordatas["CAM_BACK_RIGHT"])
        else:
            raise NotImplementedError
    # qcam = torch.stack(qcam, dim=0) # [ncam,3,h,w]
    qcam = torch.cat(qcam, dim=-1)  # [3,h,w*ncam] panorama cat
    # TODO: support multi-cam setting
    # assert qcam.shape[0] == 1
    # qcam = qcam[0]

    # qcam = torch.stack([sensordatas['CAM_FRONT'],
    #                     sensordatas['CAM_FRONT_LEFT'],
    #                     sensordatas['CAM_FRONT_RIGHT'],
    #                     sensordatas['CAM_BACK'],
    #                     sensordatas['CAM_BACK_LEFT'],
    #                     sensordatas['CAM_BACK_RIGHT']], dim=0) # [ncam,3,h,w]

    return qcam, sensordatas["RANGE_DATA"], sensordatas["BEV_DATA"], sensordatas["LIDAR_TOP"]


def get_seq_sample_tokens(nusc, sampletoken, seq_len, current_frame_type):
    # current_frame_type: ['new', 'mid', 'old']
    sample = nusc.get("sample", sampletoken)
    seq_sample_tokens = []
    if current_frame_type == "new":
        for _i in range(seq_len):
            seq_sample_tokens = [sample["token"], *seq_sample_tokens]
            sampletoken = sample["prev"]
            if len(sampletoken) == 0:
                sampletoken = sample["token"]
            sample = nusc.get("sample", sampletoken)

    elif current_frame_type == "old":
        for _i in range(seq_len):
            seq_sample_tokens.append(sample["token"])
            sampletoken = sample["next"]
            if len(sampletoken) == 0:
                sampletoken = sample["token"]
            sample = nusc.get("sample", sampletoken)

    elif current_frame_type == "mid":
        midsample = sample
        seq_sample_tokens.append(sample["token"])  # middle one
        for _i in range(seq_len // 2):
            sampletoken = sample["prev"]
            if len(sampletoken) == 0:
                sampletoken = sample["token"]
            sample = nusc.get("sample", sampletoken)
            seq_sample_tokens = [sample["token"], *seq_sample_tokens]
        sample = midsample
        for _i in range(seq_len // 2):
            sampletoken = sample["next"]
            if len(sampletoken) == 0:
                sampletoken = sample["token"]
            sample = nusc.get("sample", sampletoken)
            seq_sample_tokens.append(sample["token"])

    assert len(seq_sample_tokens) == seq_len
    return seq_sample_tokens
