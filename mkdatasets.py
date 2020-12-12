import os
import shutil
import sys
from PIL import Image

SIZE = 180

def mkds(name, sets, in_path, out_path):
    """
    sets looks like {new_name: [old1, old2, old3, ...]}
    """
    os.makedirs(f"{out_path}/{name}")
    for new_class,members in sets.items():
        os.makedirs(f"{out_path}/{name}/{new_class}")
        for s in members:
            for fname in os.listdir(f"{in_path}/{s}"):
                im = Image.open(f"{in_path}/{s}/{fname}")
                im = im.resize((SIZE, SIZE))
                im.save(f"{out_path}/{name}/{new_class}/{fname}")

def main():
    if len(sys.argv) != 3:
        print(f"Expected usage: {sys.argv[0]} <Input path> <Datasets path>")
        return

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    os.makedirs(out_path)
    mkds("game_ident", {
        "warships": ["warships", "warships_game", "warships_menu"],
        "warzone": ["warzone", "warzone_game", "warzone_gulag", "warzone_menu"],
        "obduction": ["obduction"],
        "overwatch": ["overwatch", "overwatch_game", "overwatch_menu"],
        "blank": ["blank"]
    }, in_path, out_path)
    mkds("warzone", {"game": ["warzone_game", "warzone_gulag"], "menu": ["warzone_menu"], "blank": ["blank"]}, in_path, out_path)
    mkds("overwatch", {"game": ["overwatch_game"], "menu": ["overwatch_menu"], "blank": ["blank"]}, in_path, out_path)
    mkds("warships", {"game": ["warships_game"], "menu": ["warships_menu"], "blank": ["blank"]}, in_path, out_path)
    pass

if __name__ == "__main__":
    main()
