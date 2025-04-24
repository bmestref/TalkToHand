import os, json, zipfile, glob, subprocess
import tensorflow as tf


def load_dataset(
    kaggle_username: str,
    kaggle_key: str,
    dataset_id: str,
    extract_to: str = "data",
    image_size: tuple = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 123):

    # Step 1: Setup Kaggle API
    os.makedirs("/root/.kaggle", exist_ok=True)
    with open("/root/.kaggle/kaggle.json", "w") as f:
        json.dump({"username": kaggle_username, "key": kaggle_key}, f)
    os.chmod("/root/.kaggle/kaggle.json", 0o600)

    # Step 2: Install Kaggle API
    subprocess.run("pip install -q kaggle", shell=True)

    # Step 3: Download dataset
    subprocess.run(f"kaggle datasets download -d {dataset_id}", shell=True)

    # Step 4: Unzip dataset
    zip_file = glob.glob("*.zip")[0]
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # Step 5: Locate dataset folder
    data_root = os.path.join(extract_to, next(os.walk(extract_to))[1][0])

    # Step 6: Load tf.data.Dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_root,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_root,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    # Step 7: Optional performance boost
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds
    