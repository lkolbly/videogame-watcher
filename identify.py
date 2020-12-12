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

#FRAMESKIP = 60 * 10 # 10s

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

class MenuGameTagger:
    def __init__(self, name, video, args):
        """import tensorflow as tf

        def load_model(path, class_names):
            model = tf.keras.models.load_model(path)
            def runmodel(arr):
                predictions = model.predict(tf.expand_dims(arr, 0))
                score = tf.nn.softmax(predictions[0])
                return (class_names[np.argmax(score)], 100 * np.max(score))
            return runmodel"""

        self.model = load_model(f"{args.models_path}/{name}.h5")
        self.video = video
        self.retrain_path = args.retrain_path
        self.last_state = None
        self.segments = []
        self.name = name
        #os.makedirs(f"{name}-out", exist_ok=True)
        self.retrain = {}

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
            shutil.move(fname, f"retrain/{h}.png")

        return

        fname_base = video.split("/")[-1].split(".")[0]
        with open(f"{self.name}-out/{fname_base}.json", "w") as f:
            f.write(json.dumps(self.segments))
        return

        fullclip = VideoFileClip(video)

        match_count = 0
        for idx in range(len(self.segments) - 1):
            if self.segments[idx][1] != "game":
                continue
            start = self.segments[idx][0]
            end = self.segments[idx+1][0]
            print("{} game starting at {} lasting for {}".format(self.name, frames_to_ts(start), frames_to_ts(end - start)))
            subclip = fullclip.subclip(frames_to_ts(start), frames_to_ts(end))
            subclip.write_videofile(f"{self.name}-out/{fname_base}-{match_count}.mp4", codec="mpeg4", audio_bitrate="48k", bitrate="12000k")
            match_count += 1
            pass
        pass

def process_video(video, args, use_bar=False):
    if not video.endswith(".flv"):
        return

    ident_model = load_model(f"{args.models_path}/ident.h5")
    #warzone_model = load_model("warzone.h5", ["game", "menu"])

    #find_unident_frames(ident_model, video, "warzone", use_bar)
    #return

    cap = cv2.VideoCapture(video)

    #possible_retrain = {}

    frameno = 0
    #segments = []
    #last_game = None
    taggers = {}
    for game in ["warzone", "overwatch", "warships"]:
        taggers[game] = MenuGameTagger(game, video, args)
    games = collections.Counter()
    print("Processing {}...".format(video))
    #with tqdm() as bar:
    force_retrain = {}
    #possible_retrains = {}
    if use_bar:
        bar = tqdm()
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
        #(pred, score) = warzone_model(arr)
        #print("Frame: {} FPS: {:.02} Game: {} Class: {} Confidence: {}%".format(frameno, 1.0 / max(0.000001, time.time() - start), pred_game, pred, score))
        #bar.update(FRAMESKIP)
        if use_bar:
            bar.update(args.skip_frames)

        if pred_score < 80:# or score < 80 or pred_game == "obduction":
            #h = hashlib.sha224(im_orig.resize((256, 256)).tobytes()).hexdigest()
            #print(f"Retraining {h}")
            #r,g,b = im_orig.split()
            #im_swapped = PIL.Image.merge("RGB", (b, g, r))
            #im_swapped.save(f"retrain/{h}.png")
            pass
        else:
            games[pred_game] += 1

        """(pred, score) = warzone_model(arr)
        if pred != last_game:
            last_game = pred
            segments.append((frameno, pred))"""
        #h = hashlib.sha224(im_orig.resize((256, 256)).tobytes()).hexdigest()
        #print(f"Retraining {h}")
        #r,g,b = im_orig.split()
        #im_swapped = PIL.Image.merge("RGB", (b, g, r))
        #im_swapped.save(f"retrain/{h}.png")
        #possible_retrain.append((h, im_swapped))
        #possible_retrain[h] = im_swapped
        for tagger in taggers.values():
            tagger.process(frameno, im)
        """if score < 80:
            h = hashlib.sha224(im_orig.resize((256, 256)).tobytes()).hexdigest()
            print(f"Retraining {h}")
            r,g,b = im_orig.split()
            im_swapped = PIL.Image.merge("RGB", (b, g, r))
            #im_swapped.save(f"retrain/{h}.png")
            #possible_retrain.append((h, im_swapped))
            possible_retrain[h] = im_swapped"""
        #if pred_score < 80 and len(force_retrain) < 50:
        #    force_retrain[h] = im_swapped

        #predictions = warzone_model.predict(tf.expand_dims(arr, 0))
        #score = tf.nn.softmax(predictions[0])
        #print("Frame: {} FPS: {:.02} Class: {} Confidence: {}%".format(frameno, 1.0 / max(0.000001, time.time() - start), class_names[np.argmax(score)], 100 * np.max(score)))
        pass

    cap.release()

    fname_base = video.split("/")[-1].split(".")[0]
    #with open(f"ident-out/{fname_base}.json", "w") as f:
    #    f.write(json.dumps(games))

    print("Game counts for {}: {}".format(video, games))
    if len(games.most_common()) == 0:
        for tagger in taggers:
            tagger.discard()
        return

    game = games.most_common(1)[0][0]
    #if game not in taggers:
    #    return

    #taggers[game][0].finish(video)
    #for h,im in taggers[game][1].items():
    #    #im.save(f"retrain/{h}.png")
    #    shutil.move(im, f"retrain/{h}.png")

    for gname in taggers.keys():
        if gname != game:
            taggers[gname].discard()
        else:
            taggers[gname].finish()
            #for h,im in taggers[gname][1].items():
            #    os.remove(im)

    if args.known_game_path is not None:
        find_unident_frames(ident_model, video, game, args, use_bar)

    #for h,im in force_retrain.items():
    #    im.save(f"retrain/{h}.png")

    """if games.most_common(1)[0][0] == "warzone":
        with open(f"warzone-out/{fname_base}.json", "w") as f:
            f.write(json.dumps(segments))

        for h,im in possible_retrain.items():
            im.save(f"retrain/{h}.png")"""

    #return games.most_common(1)[0][0]

def find_unident_frames(ident_model, video, classification, args, use_bar=False):
    os.makedirs(f"retrain-{classification}", exist_ok=True)
    #ident_model = load_model("ident.h5", ["blank", "obduction", "overwatch", "warships", "warzone"])
    #warzone_model = load_model("warzone.h5", ["game", "menu"])

    cap = cv2.VideoCapture(video)

    #possible_retrain = {}

    frameno = 0
    #segments = []
    #last_game = None
    #taggers = {}
    #for game in ["warzone", "overwatch", "warships"]:
    #    taggers[game] = (MenuGameTagger(game), {})
    #games = collections.Counter()
    print("Processing {}...".format(video))
    #with tqdm() as bar:
    if use_bar:
        bar = tqdm()
    #force_retrain = {}
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

        #if (pred_score < 80 and pred_game != "blank") or pred_game not in [classification, "blank"]:
        if (pred_score < 80 or pred_game != classification) and pred_game != "blank":
            im_swapped.save(f"{args.known_game_path}/{classification}/{h}.png")

    cap.release()

    """fname_base = video.split("/")[-1].split(".")[0]
    with open(f"ident-out/{fname_base}.json", "w") as f:
        f.write(json.dumps(games))

    print("Game counts for {}: {}".format(video, games))
    if len(games.most_common()) == 0:
        return

    game = games.most_common(1)[0][0]
    if game not in taggers:
        return

    taggers[game][0].finish(video)
    for h,im in taggers[game][1].items():
        im.save(f"retrain/{h}.png")"""
    pass

"""def process_video(video):
    cap = cv2.VideoCapture(video)

    frameno = 0
    segments = []
    last_game = None
    #with tqdm() as bar:
    while cap.isOpened():
        frameno += 1
        start = time.time()
        ret, frame = cap.read()
        if frameno % FRAMESKIP != 0:
            # Only pull every 10s
            continue
        if frame is None:
            break
        im_orig = PIL.Image.fromarray(frame)
        im = im_orig.resize((180, 180))
        arr = np.array(im)
        (pred_game, pred_score) = ident_model(arr)
        (pred, score) = warzone_model(arr)
        if pred != last_game:
            last_game = pred
            segments.append((frameno, pred))
        #print("Frame: {} FPS: {:.02} Game: {} Class: {} Confidence: {}%".format(frameno, 1.0 / max(0.000001, time.time() - start), pred_game, pred, score))
        #bar.update(FRAMESKIP)

        # We know apriori this is a Warzone video, so...
        #if pred_score < 80 or score < 80 or pred_game == "obduction":
        if score < 80:
            h = hashlib.sha224(im_orig.resize((256, 256)).tobytes()).hexdigest()
            print(f"Retraining {h}")
            r,g,b = im_orig.split()
            im_swapped = PIL.Image.merge("RGB", (b, g, r))
            im_swapped.save(f"retrain/{h}.png")
        #predictions = warzone_model.predict(tf.expand_dims(arr, 0))
        #score = tf.nn.softmax(predictions[0])
        #print("Frame: {} FPS: {:.02} Class: {} Confidence: {}%".format(frameno, 1.0 / max(0.000001, time.time() - start), class_names[np.argmax(score)], 100 * np.max(score)))
        pass

    cap.release()

    fname_base = video.split("/")[-1].split(".")[0]
    with open(f"warzone-out/{fname_base}.json", "w") as f:
        f.write(json.dumps(segments))"""

def frames_to_ts(nframes, fps=60):
    hours = nframes // (fps * 3600)
    nframes -= hours * (fps * 3600)
    minutes = nframes // (fps * 60)
    nframes -= minutes * (fps * 60)
    seconds = nframes // fps
    nframes -= seconds * fps
    milliseconds = int(nframes / fps * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

if __name__ == "__main__":
    """
    Things the user can do to files:
    - Identify a video
      - Optionally move into labelled directory
    - Split a video into matches
    - Generate moviepy script to access matches from source (and output low-bitrate copy for editing)
    - Find uncertain frames for each model
    - Find incorrectly classified frames for videos of known content
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="Path to the models directory", required=True)

    parser.add_argument("--no-bar", help="Disable printing the progress bar", action='store_true')

    # Arguments for finding files
    parser.add_argument("--file", help="Path to a single video file to process")

    # Arguments for defining how to process found files
    parser.add_argument("--skip-frames", default="600", help="Only process every Nth frame")
    parser.add_argument("--identified-directory", help="If a file is identified, move it to the named directory")
    parser.add_argument("--retrain-directory", help="If specified, add uncertain frames to this directory")
    parser.add_argument("--known-game-directory", help="If specified, videos are re-processed using the known game identity to find frames which the ident model mis-classified")

    cmdline = parser.parse_args()

    #print(cmdline.models)

    args = Args()
    args.models_path = cmdline.models
    args.retrain_path = cmdline.retrain_directory
    args.known_game_path = cmdline.known_game_directory
    args.skip_frames = int(cmdline.skip_frames)

    process_video(cmdline.file, args, use_bar=not cmdline.no_bar)

    #process_video('/home/lane/Downloads/test-1605564339.flv')

    """with open("warzone-out/test-1605564339.json") as f:
        import json
        data = json.loads(f.read())

    print(data)

    fullclip = VideoFileClip("/home/lane/Downloads/test-1605564339.flv")

    for idx in range(len(data) - 1):
        if data[idx][1] != "game":
            continue
        start = data[idx][0]
        end = data[idx+1][0]
        print("Game starting at {} lasting for {}".format(frames_to_ts(start), frames_to_ts(end - start)))
        subclip = fullclip.subclip(frames_to_ts(start), frames_to_ts(end))
        subclip.write_videofile(f"out/{idx}.mp4", codec="mpeg4", audio_bitrate="48k", bitrate="12000k")
        pass"""

    """os.makedirs("retrain")
    os.makedirs("ident-out")

    #process_video('/home/lane/Downloads/test-1605564339.flv', use_bar=True)

    with Pool(5) as p:
        files = list(map(lambda fname: f"/data/{fname}", os.listdir("/data")))
        p.map(process_video, files)"""

    #for fname in os.listdir("/data"):
    #    fname = f"/data/{fname}"
    #    ident_video(fname)
    #    #if ident_video(fname) == "warzone":
    #    #    process_video(fname)
