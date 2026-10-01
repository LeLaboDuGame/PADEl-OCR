from datasets import load_dataset
import numpy as np
import random
import struct
from PIL import ImageOps
import time


# ---------------- CONFIG ----------------
TEST = "./letter_recognition/data/test-00000-of-00001-bc8b28dacaaa708d.parquet"
TRAIN = "./letter_recognition/data/train-00000-of-00001-92b9aa4d471d61ab.parquet"
PARQUET_FILE = TRAIN
SAVE_FILE = "dataset.bin"

# Donne ici les labels que tu veux et combien d'elements par label.
# Exemple : 100 random du label 0, puis 100 random du label 1, etc.
# Labels : 0=A, 1=B, ..., 25=Z
LABEL_COUNTS = {i: 1000 for i in range(26)}
# Exemple pour seulement 0 et 1 :
# LABEL_COUNTS = {0: 100, 1: 100}

# Seed pour le tirage aleatoire (reproductible). Mets None pour un tirage
# different a chaque execution.
SEED = int(time.time())
# ----------------------------------------


def preprocess(image):
    """Noir et blanc (binaire 0/1) + inversion, -> vecteur 784 (converti en float a l'ecriture)."""
    # 'L' = niveaux de gris 0-255, resize au cas ou
    img = image.convert("L").resize((28, 28))
    # inversion des couleurs : 255 - pixel
    img = ImageOps.invert(img)
    arr = np.array(img, dtype=np.uint8)
    # binarisation : seuil a 128 -> 0 (noir) / 1 (blanc)
    binary = (arr > 128).astype(np.int32)
    return binary.flatten().tolist()  # 784 int, ordre ligne-major


def main():
    dataset = load_dataset("parquet", data_files=PARQUET_FILE)["train"]

    total = sum(LABEL_COUNTS.values())
    print(f"Labels demandes : {LABEL_COUNTS}")
    print(f"Total echantillons : {total}")

    # compteurs : combien deja pris par label
    taken = {label: 0 for label in LABEL_COUNTS}
    needed = dict(LABEL_COUNTS)

    with open(SAVE_FILE, "wb") as f:
        # header : {nombre de data en unsigned long int} = uint64 little-endian (8 octets)
        f.write(struct.pack("<Q", total))

        # 1ere passe : on collecte TOUS les indices par label
        by_label_idx = {label: [] for label in needed}
        for idx, row in enumerate(dataset):
            lbl = int(row["label"])
            if lbl in needed:
                by_label_idx[lbl].append(idx)

        # tirage aleatoire : N indices random par label
        rng = random.Random(SEED)
        by_label_sampled = {}
        for label in needed:
            dispo = by_label_idx[label]
            n = needed[label]
            if len(dispo) < n:
                print(f"WARNING: label {label} : demande {n}, "
                      f"trouve seulement {len(dispo)} dans le parquet.")
                by_label_sampled[label] = dispo
            else:
                by_label_sampled[label] = rng.sample(dispo, n)

        # 2eme passe : on melange TOUT (pas 100 A puis 100 B cote a cote)
        # liste globale (label, idx) puis shuffle
        all_picks = []
        for label in needed.keys():
            for idx in by_label_sampled[label]:
                all_picks.append((label, idx))
        rng.shuffle(all_picks)

        written = 0
        for label, idx in all_picks:
                img = dataset[idx]["image"]
                flat = preprocess(img)  # liste de 784 int (0/1)
                # one-hot 26 cases : 1 a la case de la lettre, 0 ailleurs
                one_hot = [0.0] * 26
                one_hot[int(label)] = 1.0
                f.write(struct.pack("<26f", *one_hot))  # 26 x float32
                # {flatten image 28x28} = 784 x float32 little-endian (0.0/1.0)
                f.write(struct.pack("<784f", *[float(x) for x in flat]))
                taken[label] += 1
                written += 1

        print(f"Ecrit {written}/{total} echantillons dans {SAVE_FILE}")

    # rappel du format binaire pour lecture en C :
    #   unsigned long nb;  fread(&nb, 8, 1, f);   // '<Q'
    #   loop nb fois :
    #     float label[26]; fread(label, 4, 26, f);  // one-hot float, label[i]==1.0 => lettre i
    #     float px[784];   fread(px, 4, 784, f);    // 784 x '<f', valeurs 0.0/1.0


if __name__ == "__main__":
    main()
