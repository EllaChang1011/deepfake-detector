import os
import random

def split_real_fake_dirs(real_root, fake_root, test_fake_root=None, seed=42):
    random.seed(seed)
    real_all = sorted(os.listdir(real_root))
    fake_all = sorted(os.listdir(fake_root))

    n = len(real_all)
    train_n = int(0.8 * n)
    val_n = int(0.1 * n)

    random.shuffle(real_all)
    real_train = [os.path.join(real_root, d) for d in real_all[:train_n]]
    real_val   = [os.path.join(real_root, d) for d in real_all[train_n:train_n+val_n]]
    real_test  = [os.path.join(real_root, d) for d in real_all[train_n+val_n:]]

    fake_train = [os.path.join(fake_root, d) for d in fake_all]
    fake_test = []
    if test_fake_root:
        fake_test = [os.path.join(test_fake_root, d) for d in os.listdir(test_fake_root)]

    return {
        "train": (real_train, fake_train),
        "val": (real_val, []),
        "test": (real_test, fake_test)
    }
