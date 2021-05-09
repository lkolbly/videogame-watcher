import PIL
import time
import numpy as np
import cv2
import os
import hashlib
import collections
from tqdm import tqdm
import json
from multiprocessing import Pool
from moviepy.editor import *
import shutil
import tempfile
import re
import datetime

def load_model(path):
    class_names = json.loads(open(f"{path}.json").read())

    import silence_tensorflow.auto
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    def runmodel(arr):
        predictions = model.predict(tf.expand_dims(arr, 0))
        score = tf.nn.softmax(predictions[0])
        return (class_names[np.argmax(score)], 100 * np.max(score))
    return runmodel

class Args:
    def __init__(self):
        self.models_path = None
        self.retrain_path = None
        self.known_game_path = None
        self.skip_frames = None
        self.moviepy = None
        self.identified_path = None
        self.youtube_key = None

class MenuGameTagger:
    def __init__(self, name, video, args):
        self.model = load_model(f"{args.models_path}/{name}.h5")
        self.video = video
        self.retrain_path = args.retrain_path
        self.last_state = None
        self.segments = []
        self.name = name
        self.retrain = {}
        self.moviepy = args.moviepy
        self.matches = []

        if self.retrain_path is not None:
            os.makedirs(self.retrain_path, exist_ok=True)

    def process(self, frameno, im):
        """
        frame is the converted/preprocessed 180x180x3 numpy array
        Returns True if the frame should be saved for retraining
        """
        frame = np.array(im)
        (pred, score) = self.model(frame)
        should_retrain = score < 80
        if pred == "blank":
            return should_retrain

        if pred != self.last_state:
            self.last_state = pred
            self.segments.append((frameno, pred))

        if self.retrain_path is not None and score < 80:
            h = hashlib.sha224(im.resize((256, 256)).tobytes()).hexdigest()
            r,g,b = im.split()
            im_swapped = PIL.Image.merge("RGB", (b, g, r))

            fd,tmpfname = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            im_swapped.save(tmpfname)
            self.retrain[h] = tmpfname

        return should_retrain

    def discard(self):
        for h,fname in self.retrain.items():
            os.remove(fname)

    def finish(self):
        for h,fname in self.retrain.items():
            shutil.move(fname, f"{self.retrain_path}/{h}.png")

        match_count = 0
        for idx in range(len(self.segments) - 1):
            if self.segments[idx][1] != "game":
                continue
            start = self.segments[idx][0]
            end = self.segments[idx+1][0]
            if end - start < 30*60: # A match has to be at least 30 seconds to count
                continue
            self.matches.append((start, end))
            match_count += 1

        if self.moviepy is not None:
            fname_base = self.video.split("/")[-1].split(".")[0]

            os.makedirs(self.moviepy, exist_ok=True)
            fullclip = VideoFileClip(self.video)
            fullclip.resize((1920/2, 1080/2)).write_videofile(f"{self.moviepy}/{fname_base}.mp4", codec="mpeg4", audio_bitrate="48k", bitrate="3000k")

            with open(f"{self.moviepy}/{fname_base}.py", "w") as f:
                f.write(f"# {self.name} matches")
                f.write("from moviepy.editor import *\n\n")
                f.write(f"fullclip = VideoFileClip(\"{fname_base}\".mp4)\n")

                for match_count, (start, end) in enumerate(self.matches):
                    f.write(f"match{match_count} = fullclip.subclip(\"{frames_to_ts(start)}\", \"{frames_to_ts(end)}\")\n")
                    pass

                f.write("\n")
                for i in range(match_count):
                    f.write(f"match{i}.write_videofile(\"match{i}.mp4\", codec=\"mpeg4\", audio_bitrate=\"48k\")\n")
            pass

def process_video(video, args, use_bar=False):
    if not video.endswith(".flv"):
        return

    ident_model = load_model(f"{args.models_path}/ident.h5")

    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frameno = 0
    taggers = {}
    for game in ["warzone", "overwatch", "warships"]:
        taggers[game] = MenuGameTagger(game, video, args)
    games = collections.Counter()
    print(f"Processing {fps}fps {frame_count}-frame file {video}...")
    force_retrain = {}
    if use_bar:
        bar = tqdm(total=frame_count)
    while cap.isOpened():
        frameno += 1
        start = time.time()
        ret, frame = cap.read()
        if frameno % args.skip_frames != 0:
            # Only pull every 10s
            continue
        if frame is None:
            break
        im_orig = PIL.Image.fromarray(frame)
        im = im_orig.resize((180, 180))
        arr = np.array(im)
        (pred_game, pred_score) = ident_model(arr)
        if use_bar:
            bar.update(args.skip_frames)

        if pred_score < 80:
            pass
        else:
            games[pred_game] += 1

        for tagger in taggers.values():
            tagger.process(frameno, im)
    if use_bar:
        bar.close()

    cap.release()

    print("Game counts for {}: {}".format(video, games))
    if len(games.most_common()) == 0:
        for tagger in taggers.values():
            tagger.discard()
        return

    if args.assume_game is not None:
        game = args.assume_game
    else:
        game = games.most_common(1)[0][0]

    tagger = None
    for gname in taggers.keys():
        if gname != game:
            taggers[gname].discard()
        else:
            taggers[gname].finish()
            tagger = taggers[gname]

    if args.known_game_path is not None:
        find_unident_frames(ident_model, video, game, args, use_bar)

    if args.youtube_key is not None and tagger is not None:
        print("Generating YouTube description")
        #print("Matches: ", tagger.matches)

        description = "{} match {}\n\n".format(game, video.split("/")[-1])
        description += "0:00 Start\n"
        for i,match in enumerate(tagger.matches):
            description += "{} Match {}\n".format(frames_to_ts_minsec(match[0]), i)
        description += "\nAutomatically uploaded by Game Replay Uploader"

        m = re.search(r"test-(\d+)\.", video)
        game_pretty_name = game.capitalize()
        if m is not None:
            dt = datetime.datetime.fromtimestamp(int(m.group(1)))
            title = "{} game {}".format(game_pretty_name, dt.strftime("%c"))
        else:
            title = "{} Game Unknown Date: {}".format(game_pretty_name, video)
        print(title)
        print(description)

    fname = video.split("/")[-1]
    if args.identified_path is not None:
        os.makedirs(f"{args.identified_path}/{game}", exist_ok=True)
        shutil.move(video, f"{args.identified_path}/{game}/{fname}")

def find_unident_frames(ident_model, video, classification, args, use_bar=False):
    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frameno = 0
    if use_bar:
        bar = tqdm(total=frame_count)
    while cap.isOpened():
        frameno += 1
        start = time.time()
        ret, frame = cap.read()
        if frameno % args.skip_frames != 0:
            # Only pull every 10s
            continue
        if frame is None:
            break
        im_orig = PIL.Image.fromarray(frame)
        im = im_orig.resize((180, 180))
        arr = np.array(im)
        (pred_game, pred_score) = ident_model(arr)

        if use_bar:
            bar.update(args.skip_frames)

        h = hashlib.sha224(im_orig.resize((256, 256)).tobytes()).hexdigest()
        r,g,b = im_orig.split()
        im_swapped = PIL.Image.merge("RGB", (b, g, r))

        if (pred_score < 80 or pred_game != classification) and pred_game != "blank":
            im_swapped.save(f"{args.known_game_path}/{classification}/{h}.png")
    if use_bar:
        bar.close()

    cap.release()

def frames_to_ts(nframes, fps=60):
    hours = nframes // (fps * 3600)
    nframes -= hours * (fps * 3600)
    minutes = nframes // (fps * 60)
    nframes -= minutes * (fps * 60)
    seconds = nframes // fps
    nframes -= seconds * fps
    milliseconds = int(nframes / fps * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def frames_to_ts_minsec(nframes, fps=60):
    hours = nframes // (fps * 3600)
    nframes -= hours * (fps * 3600)
    minutes = nframes // (fps * 60)
    nframes -= minutes * (fps * 60)
    seconds = nframes // fps
    nframes -= seconds * fps
    milliseconds = int(nframes / fps * 1000)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes:02}:{seconds:02}"

if __name__ == "__main__":
    """
    Things the user can do to files:
    - Identify a video
      - Optionally move into labelled directory
    - Split a video into matches
    - Generate moviepy script to access matches from source (and output low-bitrate copy for editing)
    - Find uncertain frames for each model
    - Find incorrectly classified frames for videos of known content
    - Upload match-ified videos to YouTube
    """

    #import argparse

    #parser = argparse.ArgumentParser()
    parser = argparser
    parser.add_argument("--models", help="Path to the models directory", required=True)

    parser.add_argument("--no-bar", help="Disable printing the progress bar", action='store_true')

    # Arguments for finding files
    parser.add_argument("--file", help="Path to a single video file to process")
    parser.add_argument("--input-path", help="Path to a folder containing video files to process")

    # Arguments for defining how to process found files
    parser.add_argument("--skip-frames", default="600", help="Only process every Nth frame")
    parser.add_argument("--identified-directory", help="If a file is identified, move it to the named directory")
    parser.add_argument("--retrain-directory", help="If specified, add uncertain frames to this directory")
    parser.add_argument("--known-game-directory", help="If specified, videos are re-processed using the known game identity to find frames which the ident model mis-classified")
    parser.add_argument("--moviepy-out", help="Target directory to save moviepy scripts and low-res working sets")
    parser.add_argument("--youtube-key", help="API key for uploading to YouTube")
    parser.add_argument("--assume-game", help="If set, override the detected game with the provided value")

    cmdline = parser.parse_args()

    args = Args()
    args.models_path = cmdline.models
    args.retrain_path = cmdline.retrain_directory
    args.known_game_path = cmdline.known_game_directory
    args.skip_frames = int(cmdline.skip_frames)
    args.moviepy = cmdline.moviepy_out
    args.identified_path = cmdline.identified_directory
    args.youtube_key = cmdline.youtube_key
    args.assume_game = cmdline.assume_game

    if cmdline.file is not None and cmdline.input_path is not None:
        raise RuntimeError("Cannot specify both --file and --input-path")

    if cmdline.file is not None:
        process_video(cmdline.file, args, use_bar=not cmdline.no_bar)
    else:
        for fname in sorted(os.listdir(cmdline.input_path)):
            fname = f"{cmdline.input_path}/{fname}"
            process_video(fname, args, use_bar=not cmdline.no_bar)

    """os.makedirs("retrain")
    os.makedirs("ident-out")

    #process_video('/home/lane/Downloads/test-1605564339.flv', use_bar=True)

    with Pool(5) as p:
        files = list(map(lambda fname: f"/data/{fname}", os.listdir("/data")))
        p.map(process_video, files)"""
