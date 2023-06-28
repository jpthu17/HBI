from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from os.path import exists
import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from .rawvideo_util import RawVideoExtractor
import torch
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, ToPILImage, ColorJitter, RandomHorizontalFlip, RandomResizedCrop


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def takeFirst(item):
    return item[0]

class MSRVTT_DataLoader(Dataset):
    """MSRVTT test dataset loader."""
    def __init__(
            self,
            jsonl_path,
            train_jsonl,
            ans2label_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            use_num=1000,
    ):
        self.csv = load_jsonl(jsonl_path)
        self.image_resolution = image_resolution
        self.ans2label = load_json(ans2label_path)
        self.num_labels = use_num
        self.label2ans = {v: k for k, v in self.ans2label.items()}

        ans_count = {k: 0 for k, v in self.ans2label.items()}
        train_dataset = load_jsonl(train_jsonl)
        for i in train_dataset:
            ans_count[i["answer"]] += 1
        sorted = [[v, k] for k, v in ans_count.items()]
        sorted.sort(key=takeFirst, reverse=True)
        self.use_ans2label = {}
        for i in range(use_num):
            self.use_ans2label[sorted[i][1]] = i

        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            self.sentences_dict = {}
            for itm in self.csv:
                if itm['answer'] in self.ans2label:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['question'], itm['answer'])
            self.sample_len = len(self.sentences_dict)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)

        self.transform = Compose([
            Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo_dec(self, video_id, s=None, e=None):
        # speed up video decode via decord.
        video_mask = np.zeros(self.max_frames, dtype=np.long)
        max_video_length = 0

        # T x 3 x H x W
        video = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1
        video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))

        if exists(video_path):
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            print(video_path)
            raise FileNotFoundError

        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        if num_frames > 0:
            # T x 3 x H x W
            sample_fps = int(self.feature_framerate)
            t_stride = int(round(float(fps) / sample_fps))

            all_pos = list(range(f_start, f_end + 1, t_stride))
            if len(all_pos) > self.max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
            else:
                sample_pos = all_pos

            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
            patch_images = torch.stack([self.transform(img) for img in patch_images])
            slice_len = patch_images.shape[0]
            max_video_length = max_video_length if max_video_length > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = patch_images
        else:
            print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption, answer = self.sentences_dict[idx]
        else:
            pass
            # video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo_dec(choice_video_ids[0])
        label = self.ans2label[answer] if answer in self.ans2label else -1
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, label


class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            jsonl_path,
            ans2label_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            use_num=1000,
    ):
        self.csv = load_jsonl(jsonl_path)
        self.image_resolution = image_resolution
        self.ans2label = load_json(ans2label_path)
        self.num_labels = use_num

        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0

        ans_count = {k: 0 for k, v in self.ans2label.items()}
        for i in self.csv:
            ans_count[i["answer"]] += 1
        sorted = [[v, k] for k, v in ans_count.items()]
        sorted.sort(key=takeFirst, reverse=True)
        self.use_ans2label = {}
        for i in range(use_num):
            self.use_ans2label[sorted[i][1]] = i

        if self.unfold_sentences:
            self.sentences_dict = {}
            for itm in self.csv:
                self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['question'], itm['answer'])
            self.sample_len = len(self.sentences_dict)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)

        self.transform = Compose([
            Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo_dec(self, video_id, s=None, e=None):
        # speed up video decode via decord.
        video_mask = np.zeros(self.max_frames, dtype=np.long)
        max_video_length = 0

        # T x 3 x H x W
        video = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1
        video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))

        if exists(video_path):
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            print(video_path)
            raise FileNotFoundError

        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        if num_frames > 0:
            # T x 3 x H x W
            sample_fps = int(self.feature_framerate)
            t_stride = int(round(float(fps) / sample_fps))

            all_pos = list(range(f_start, f_end + 1, t_stride))
            if len(all_pos) > self.max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
            else:
                sample_pos = all_pos

            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
            patch_images = torch.stack([self.transform(img) for img in patch_images])
            slice_len = patch_images.shape[0]
            max_video_length = max_video_length if max_video_length > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = patch_images
        else:
            print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption, answer = self.sentences_dict[idx]
        else:
            pass
            # video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo_dec(choice_video_ids[0])
        label = self.ans2label[answer]
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, label