from paths import EMBEDDING_DIR


def get_emb_filename(side, embedding_type, is_train):
    file = f"{'train' if is_train else 'test'}_{embedding_type}_{side}.ebm"
    return EMBEDDING_DIR / file
