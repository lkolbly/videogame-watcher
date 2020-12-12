import os
import shutil

def mkds(name, sets):
    """
    sets looks like {new_name: [old1, old2, old3, ...]}
    """
    os.makedirs(f"datasets/{name}")
    for new_class,members in sets.items():
        os.makedirs(f"datasets/{name}/{new_class}")
        for s in members:
            for fname in os.listdir(f"dataset/{s}"):
                shutil.copy(f"dataset/{s}/{fname}", f"datasets/{name}/{new_class}/{fname}")

if __name__ == "__main__":
    os.makedirs("datasets")
    mkds("game_ident", {"warships": ["warships", "warships_game", "warships_menu"], "warzone": ["warzone", "warzone_game", "warzone_gulag", "warzone_menu"], "obduction": ["obduction"], "overwatch": ["overwatch", "overwatch_game", "overwatch_menu"], "blank": ["blank"]})
    mkds("warzone", {"game": ["warzone_game", "warzone_gulag"], "menu": ["warzone_menu"], "blank": ["blank"]})
    mkds("overwatch", {"game": ["overwatch_game"], "menu": ["overwatch_menu"], "blank": ["blank"]})
    mkds("warships", {"game": ["warships_game"], "menu": ["warships_menu"], "blank": ["blank"]})
    pass
