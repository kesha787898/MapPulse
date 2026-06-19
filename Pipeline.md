### ⚠️ Prerequisites
* **Python  Setup:**  `pip install -r requirements.txt`.
* **Directory Setup:** You must manually create the necessary data folders structure in your project directory.
* Or you can download them and unzip into /data with those links. Be carefull there are more than 40GB, but onlyy first three is required for the training script:
1. Tabular data and styles for map rendering :https://mega.nz/file/snBDWLAQ#Ilk7n5ux-qnsdoQ-Ri60XJmtWnq6mB0djtvsYcYmtq0
2. Embeddings https://mega.nz/file/B6gmlKDK#zykCrdo1X0gF6U3ymACeQdheZesa6iL-a8FuTqqAQHI
3. Models 
4. Mbtiles https://mega.nz/file/YiZ31Trb#7Wdk2l_FR1LA0E7cDZ1ENcGi6AnuY3KudYFJAaqvKsM
4. pretraining_images https://mega.nz/file/1iBQ1bJA#oBjSAoIpsBlkN3yEct7M__Z5mufoU8thpRCy9YMIEoI
5. training_images https://mega.nz/file/ky5AFYpQ#D5G5oPYkrr36OXfbO6IidHXfBLuPaApNOZCJ_VvPWTY
6. pbf files


* **Path Configuration:** All file paths are managed in `paths.py`. Before running any scripts, ensure these paths are correctly configured to match your local environment.

---

## 🛠 Reproduction Pipeline

Follow these steps in the exact order listed below to reproduce the results.

### Phase 1: Tabular Data Processing
1.  **Place Pulse v2:** Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shubham6147/mit-place-pulse).
2.  **Liveability Data:** Download [liveability.pdf](https://media.heraldsun.com.au/files/liveability.pdf) and convert it to CSV using:
    `python data_processing/tabular_datasets/pdf_to_csv.py`
3.  **Numbeo Data:** Download the `numbeo.csv` from [Numbeo Rankings](https://www.numbeo.com/quality-of-life/rankings.jsp).
4.  **Join Datasets:** Run the joining script:
    `python data_processing/tabular_datasets/tabular_datasets_join.py`

### Phase 2: Map Tile Rendering
8.  **Download PBF:** Run `python data_processing/image_rendering/pbf/download_raw_pbf_by_cities.py`. 
    *Note: This process is time-intensive.*
9.  **Merge & Convert:** Run the PowerShell scripts in order(you must manualy insert your paths in those scripts):
    * `data_processing/image_rendering/merge_pbf/merge.ps1`
    * `data_processing/image_rendering/merge_pbf/to_mbtiles.ps1`
10. **Start TileServer GL:** Run the Docker container to serve your tiles:
    ```bash
    docker run --rm -it \
      -v D:\projects\maps\data\mbtiles:/data \
      -v D:\projects\maps\data\raw\styles\osm-bright:/usr/src/app/node_modules/tileserver-gl-styles/styles/osm-bright \
      -p 8080:80 klokantech/tileserver-gl /data/ --verbose
    ```

### Phase 3: Image Acquisition & Dataset Preparation
11. **Pretraining Images:** Run `python data_processing/pretraining/download_images.py`
12. **Training Images:** Run `python data_processing/training/download_images.py`
13. **Full Dataset:** Run `python data_processing/tabular_datasets/full_dataset.py`
14. **Split Data:** Run `python data_processing/data_split/split_test_train.py`
15. **Generate IDs:** Run `python data_processing/data_split/create_place_ids_for_pulse.py`

### Phase 4: Training & Evaluation
16. **Pretraining:** Run `python pretraining/pretrain.py`
17. **Embeddings:** Run `python training/embeddiings/infer.py`
18. **Training:** Run `python training/train.py`
19. **Evaluation:** Run `python evaluation/eval.py`

---

## 📂 Project Structure Note
Ensure that all output directories are created manually if the scripts do not automatically generate them. Review `paths.py` periodically if you encounter "File Not Found" errors during execution.

---
