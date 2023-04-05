import shutil

import torch
print(torch.__version__)
from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights, AlexNet_Weights, VGG16_BN_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import glob
import os
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse
import json


parser = argparse.ArgumentParser(description='Data generator')
parser.add_argument('--csv_file', type=str, help='csv file')
parser.add_argument('--gpv_folder', type=str, help='csv folder')
parser.add_argument('--lavis_folder', type=str, help='output folder')
parser.add_argument('--gt_folder', type=str, help='ground truth csv folder')
parser.add_argument('--video_folder', type=str, help='video folder')
parser.add_argument('--output_folder', type=str, help='output folder')
parser.add_argument('--start_video', type=int)
parser.add_argument('--last_video', type=int)

args = parser.parse_args()

m_50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
m_152 = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
m_alex = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1)
m_vgg16_bn = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', weights=VGG16_BN_Weights.IMAGENET1K_V1)

return_nodes_resnet = {
    'avgpool': 'out',
}
return_node_alex = {
    'classifier.5': 'out'
}
return_node_vgg = {
    'classifier.5': 'out'
}

model_50 = create_feature_extractor(m_50, return_nodes=return_nodes_resnet).to("cuda:0")
model_152 = create_feature_extractor(m_152, return_nodes=return_nodes_resnet).to("cuda:0")
model_alex = create_feature_extractor(m_alex, return_nodes=return_node_alex).to("cuda:0")
model_vgg16_bn = create_feature_extractor(m_vgg16_bn, return_nodes=return_node_vgg).to("cuda:0")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform_norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean, std)
    ]
)


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def compare_images(imageA, imageB):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    return m, s


csv_folder = args.gpv_folder
lavis_folder = args.lavis_folder
gt_folder = args.gt_folder
video_parent_folder = args.video_folder
output_folder = args.output_folder
csv = args.csv_file
last_video = args.last_video + 1
start_video = args.start_video

df_measure = pd.read_csv(csv)
df_measure.head()

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder)

csv_path = os.path.join(output_folder, 'csv')
csv_t_path = os.path.join(output_folder, 'csv_t')

os.makedirs(csv_path)
os.makedirs(csv_t_path)

final_dict = {}
for vid_count in range(start_video, last_video):
    parent_video = str(vid_count).zfill(3)
    video_folder = os.path.join(video_parent_folder, f"video_{parent_video}/trimmed_video")

    k_f_folds = ["_".join(k_i.split('_')[1:]).replace('.mp4', '') for k_i in os.listdir(video_folder) if
                 'keyframes' in k_i]

    for video_name in k_f_folds:
        segment = int(video_name.split('_')[-1].replace('seg', '').replace('.mp4', ''))
        print(f"Video: {vid_count}, Segment: {segment}")

        key_frame_paths = os.path.join(
            video_folder,
            f'keyframes_{video_name}.mp4'
        )

        tn = len(os.listdir(key_frame_paths)) + 1

        val = "os.path.join(key_frame_paths, f'{video_name}_{x}.jpeg')"
        tester = f"os.path.exists({val})"

        images = [eval(val) for x in range(tn) if eval(tester)]

        if vid_count == 10 and segment == 1:
            images = images[52:72]

        out_resnet_50 = []
        out_resnet_152 = []
        out_alex = []
        out_vgg16_bn = []
        out_mse = []
        out_ssim = []
        out_feat_cos = []
        gt_feat = []
        gt_feat_cos = []
        out_feat = []
        lavis_feature = []
        lavis_feature_cos = []

        csv_file = os.path.join(
            csv_folder,
            f"video-{vid_count}-segment-{segment}.csv"
        )

        lavis_file = os.path.join(
            lavis_folder,
            f"video-{vid_count}-segment-{segment}.csv"
        )

        df = pd.read_csv(csv_file)
        df_columns = df.columns[1:]

        df_lavis = pd.read_csv(lavis_file)
        df_lavis_column = df_lavis.columns[1:]

        gt_file = os.path.join(
            gt_folder,
            f"video_{vid_count}_segment_{segment}.csv"
        )

        if not os.path.exists(gt_file):
            continue

        df_gt = pd.read_csv(gt_file)
        df_gt.dropna(how='all', axis=1, inplace=True)
        df_gt.dropna(how='all', axis=0, inplace=True)
        df_gt_columns = df_gt.columns[1:]

        # df_measure_slice = df_measure.loc[(df_measure['video'] == vid_count) & (df_measure['segment'] == segment)]
        # out_feat_measure = list(df_measure_slice['similarity-score'])
        #
        # if vid_count == 10 and segment == 1:
        #     out_feat_measure = out_feat_measure[51:70]

        for col_no in tqdm(range(len(df_gt_columns) - 1)):
            f_1_vqa = list(df_gt[df_gt_columns[col_no]])
            f_2_vqa = list(df_gt[df_gt_columns[col_no + 1]])
            col_1 = torch.FloatTensor(f_1_vqa)
            col_2 = torch.FloatTensor(f_2_vqa)
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
            gt_feat_c = cos(col_1, col_2)
            same = 0
            for r_no in range(len(f_1_vqa)):
                if str(f_1_vqa[r_no]) == str(f_2_vqa[r_no]):
                    same += 1
            gt_feat.append(same / len(f_1_vqa))
            gt_feat_cos.append(float(gt_feat_c.cpu().numpy()))

        for col_no in tqdm(range(len(df_columns) - 1)):
            f_1_vqa = list(df[df_columns[col_no]])
            f_2_vqa = list(df[df_columns[col_no + 1]])
            col_1 = torch.FloatTensor(f_1_vqa)
            col_2 = torch.FloatTensor(f_2_vqa)
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
            output_feat = cos(col_1, col_2)
            same = 0
            for r_no in range(len(f_1_vqa)):
                if str(f_1_vqa[r_no]) == str(f_2_vqa[r_no]):
                    same += 1
            out_feat.append(same / len(f_1_vqa))
            out_feat_cos.append(float(output_feat.cpu().numpy()))

        for col_no in tqdm(range(len(df_lavis_column) - 1)):
            f_1_vqa = list(df_lavis[df_lavis_column[col_no]])
            f_2_vqa = list(df_lavis[df_lavis_column[col_no + 1]])
            col_1 = torch.FloatTensor(f_1_vqa)
            col_2 = torch.FloatTensor(f_2_vqa)
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
            lav_feat = cos(col_1, col_2)
            same = 0
            for r_no in range(len(f_1_vqa)):
                if str(f_1_vqa[r_no]) == str(f_2_vqa[r_no]):
                    same += 1
            lavis_feature.append(same / len(f_1_vqa))
            lavis_feature_cos.append(float(lav_feat.cpu().numpy()))

        for i in tqdm(range(len(images) - 1)):
            image_1 = np.array(Image.open(images[i]))
            image_2 = np.array(Image.open(images[i + 1]))
            outs_50 = []
            for img in [image_1, image_2]:
                img_normalized = transform_norm(img).float()
                img_normalized = img_normalized.unsqueeze_(0)
                img_normalized = img_normalized.to("cuda:0")
                with torch.no_grad():
                    model_50.eval()
                    output_50 = model_50(img_normalized)["out"].squeeze_(0).squeeze_(1).squeeze_(1)
                    outs_50.append(output_50)

            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
            output_50 = cos(outs_50[0], outs_50[1])
            out_resnet_50.append(float(output_50.cpu().numpy()))

        for i in tqdm(range(len(images) - 1)):
            image_1 = np.array(Image.open(images[i]))
            image_2 = np.array(Image.open(images[i + 1]))
            outs_152 = []
            for img in [image_1, image_2]:
                img_normalized = transform_norm(img).float()
                img_normalized = img_normalized.unsqueeze_(0)
                img_normalized = img_normalized.to("cuda:0")
                with torch.no_grad():
                    model_152.eval()
                    output_152 = model_152(img_normalized)["out"].squeeze_(0).squeeze_(1).squeeze_(1)
                    outs_152.append(output_152)

            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
            output_152 = cos(outs_152[0], outs_152[1])
            out_resnet_152.append(float(output_152.cpu().numpy()))

        for i in tqdm(range(len(images) - 1)):
            image_1 = np.array(Image.open(images[i]))
            image_2 = np.array(Image.open(images[i + 1]))
            outs_alex = []
            for img in [image_1, image_2]:
                img_normalized = transform_norm(img).float()
                img_normalized = img_normalized.unsqueeze_(0)
                img_normalized = img_normalized.to("cuda:0")
                with torch.no_grad():
                    model_alex.eval()
                    output_alex = model_alex(img_normalized)["out"].squeeze_(0)
                    outs_alex.append(output_alex)

            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
            output_alex = cos(outs_alex[0], outs_alex[1])
            out_alex.append(float(output_alex.cpu().numpy()))

        for i in tqdm(range(len(images) - 1)):
            image_1 = np.array(Image.open(images[i]))
            image_2 = np.array(Image.open(images[i + 1]))
            outs_vgg16_bn = []
            for img in [image_1, image_2]:
                img_normalized = transform_norm(img).float()
                img_normalized = img_normalized.unsqueeze_(0)
                img_normalized = img_normalized.to("cuda:0")
                with torch.no_grad():
                    model_vgg16_bn.eval()
                    output_vgg16_bn = model_vgg16_bn(img_normalized)["out"].squeeze_(0)
                    outs_vgg16_bn.append(output_vgg16_bn)

            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
            output_vgg16_bn = cos(outs_vgg16_bn[0], outs_vgg16_bn[1])
            out_vgg16_bn.append(float(output_vgg16_bn.cpu().numpy()))

        for i in tqdm(range(len(images) - 1)):
            image_1 = np.array(Image.open(images[i]))
            image_2 = np.array(Image.open(images[i + 1]))

            k_f_img = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
            o_f_img = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
            mse_, ssim_ = compare_images(k_f_img, o_f_img)
            out_mse.append(mse_)
            out_ssim.append(ssim_)

        if vid_count not in final_dict.keys():
            final_dict[vid_count] = {}

        try:
            assert len(out_feat) == len(out_ssim)
            assert len(gt_feat_cos) == len(out_ssim)
            assert len(gt_feat) == len(gt_feat_cos)
            assert len(lavis_feature) == len(gt_feat)
            assert len(lavis_feature_cos) == len(lavis_feature)
        except Exception as e:
            print(e)
            continue

        if vid_count == 10 and segment == 1:
            final_dict[vid_count][segment] = {
                'frame pair': [f'frame-({f - 1}, {f})' for f in range(53, 53 + len(out_feat))],
                'Similarity (Human)': gt_feat,
                'gt cosine similarity': gt_feat_cos,
                'Similarity (VQA-based)': out_feat,
                'GPV VQA cosine similarity': out_feat_cos,
                "Similarity (Lavis-VQA-based)": lavis_feature,
                "Lavis VQA cosine similarity": lavis_feature_cos,
                'resnet-50 feature similarity': out_resnet_50,
                'Similarity (Feature-based)': out_resnet_152,
                'alexnet feature similarity': out_alex,
                'vgg16-bn feature similarity': out_vgg16_bn,
                'Similarity (Pixel-level)': out_ssim
            }
        else:
            final_dict[vid_count][segment] = {
                'frame pair': [f'frame-({f-1}, {f})' for f in range(1, len(out_feat) + 1)],
                'Similarity (Human)': gt_feat,
                'gt cosine similarity': gt_feat_cos,
                'Similarity (VQA-based)': out_feat,
                'GPV VQA cosine similarity': out_feat_cos,
                "Similarity (Lavis-VQA-based)": lavis_feature,
                "Lavis VQA cosine similarity": lavis_feature_cos,
                'resnet-50 feature similarity': out_resnet_50,
                'Similarity (Feature-based)': out_resnet_152,
                'alexnet feature similarity': out_alex,
                'vgg16-bn feature similarity': out_vgg16_bn,
                'Similarity (Pixel-level)': out_ssim
            }

with open("video_correlation_data.json", 'w') as f:
    f.write(json.dumps(final_dict, indent=4))

for vid in final_dict.keys():
    for seg in final_dict[vid].keys():
        data = final_dict[vid][seg]
        df = pd.DataFrame(data).T
        df.to_csv(os.path.join(csv_t_path, f'video-{vid}-segment-{seg}.csv'), header=False)

for vid in final_dict.keys():
    for seg in final_dict[vid].keys():
        data = final_dict[vid][seg]
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(csv_path, f'video-{vid}-segment-{seg}.csv'))
