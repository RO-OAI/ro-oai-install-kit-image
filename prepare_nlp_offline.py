import os
import sys
import traceback
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print(f"\n[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def safe_step(name, fn):
    print(f"\n{'=' * 80}")
    print(f"[STEP] {name}")
    print(f"{'=' * 80}")
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        traceback.print_exc()
        # Raise to make sure the process exits with non-zero if a step fails
        raise e


def prepare_dirs():
    # Use environment variable for base directory if available, otherwise default to local offline_nlp_assets
    base_str = os.environ.get("OFFLINE_NLP_DIR", str(Path.cwd() / "offline_nlp_assets"))
    base = Path(base_str)
    base.mkdir(exist_ok=True, parents=True)

    nltk_dir = base / "nltk_data"
    nltk_dir.mkdir(exist_ok=True)

    hf_home = base / "hf_home"
    hf_home.mkdir(exist_ok=True)

    gensim_dir = base / "gensim_data"
    gensim_dir.mkdir(exist_ok=True)

    os.environ["NLTK_DATA"] = str(nltk_dir)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["GENSIM_DATA_DIR"] = str(gensim_dir)

    print(f"NLTK_DATA={os.environ['NLTK_DATA']}")
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"GENSIM_DATA_DIR={os.environ['GENSIM_DATA_DIR']}")

    return base, nltk_dir, hf_home, gensim_dir


def download_nltk():
    import nltk

    # Pentru engleză și procesare standard:
    # tokenizare, stopwords, lematizare, tagging, chunking, word lists
    packages = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
    ]

    nltk_data_dir = os.environ["NLTK_DATA"]
    for pkg in packages:
        print(f"Downloading NLTK package: {pkg}")
        success = nltk.download(pkg, download_dir=nltk_data_dir)
        if not success:
            raise RuntimeError(f"Failed to download NLTK package: {pkg}")

    # Manual unzip safeguard to ensure all packages are extracted
    # NLTK's download usually unzips, but we want to be sure for offline use.
    import zipfile
    for root, dirs, files in os.walk(nltk_data_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_path = Path(root) / file
                extract_name = file[:-4]
                extract_dir = Path(root) / extract_name
                if not extract_dir.exists():
                    print(f"Manually unzipping {zip_path}")
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            # If the zip contains a single top-level folder with the same name, extract to root
                            # Otherwise, extract to extract_dir
                            top_level_dirs = {Path(n).parts[0] for n in zip_ref.namelist()}
                            if len(top_level_dirs) == 1 and list(top_level_dirs)[0] == extract_name:
                                zip_ref.extractall(root)
                            else:
                                extract_dir.mkdir(exist_ok=True)
                                zip_ref.extractall(extract_dir)
                        print(f"  [OK] Extracted to {extract_dir}")
                    except Exception as e:
                        print(f"  [WARNING] Manual unzip failed for {zip_path}: {e}")

    # Explicitly check for unzipped directories in NLTK_DATA
    # NLTK_DATA/corpora/wordnet, NLTK_DATA/corpora/omw-1.4, etc.
    nltk_data_path = Path(nltk_data_dir)
    corpora_dir = nltk_data_path / "corpora"
    if corpora_dir.exists():
        print(f"Contents of {corpora_dir}: {[p.name for p in corpora_dir.iterdir()]}")
    
    # Ensure NLTK uses our download directory
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)

    # Verification function provided by the user
    def check_nltk_resources():
        required = [
            ("tokenizers/punkt", "punkt"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
            ("corpora/words", "words"),
        ]

        missing = []
        for path, label in required:
            try:
                found = nltk.data.find(path)
                print(f"✅ Found {label} at {found}")
            except LookupError:
                missing.append(label)

        if missing:
            raise RuntimeError(f"Missing NLTK resources: {missing}")

    # Run verification
    check_nltk_resources()

    # Test rapid
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag, ne_chunk

    text = "Apple is buying a startup in London."
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    tree = ne_chunk(tags)
    sw = stopwords.words("english")
    lemma = WordNetLemmatizer().lemmatize("running", pos="v")
    
    print("NLTK tokens:", tokens[:10])
    print("NLTK stopwords sample:", sw[:5])
    print("NLTK lemma(running):", lemma)
    
    # Test WordNet and OMW
    try:
        syns = wordnet.synsets("dog")
        print("NLTK wordnet synsets for 'dog':", len(syns))
        
        # Test OMW-1.4 (multilingual)
        syns_ro = wordnet.synsets("câine", lang="ron")
        print("NLTK OMW-1.4 synsets for 'câine' (ron):", len(syns_ro))
        if len(syns_ro) == 0:
             print("[WARNING] OMW-1.4 seems empty or not working for 'ron'")
    except Exception as e:
        print(f"[ERROR] WordNet/OMW test failed: {e}")
        # List files in the corpora/wordnet to debug
        wn_dir = corpora_dir / "wordnet"
        if wn_dir.exists():
             print(f"WordNet directory exists. Contents: {[p.name for p in wn_dir.iterdir()]}")
        omw_dir = corpora_dir / "omw-1.4"
        if omw_dir.exists():
             print(f"OMW-1.4 directory exists. Contents: {[p.name for p in omw_dir.iterdir()]}")
        raise

    print("NLTK NER/chunking test OK:", tree is not None)


def download_spacy():
    # spaCy recomandă folosirea numelui complet al modelului, ex. en_core_web_sm
    run_cmd([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Apple is looking at buying a U.K. startup for $1 billion.")
    print("spaCy tokens:", [t.text for t in doc[:10]])
    print("spaCy lemmas:", [t.lemma_ for t in doc[:10]])
    print("spaCy entities:", [(ent.text, ent.label_) for ent in doc.ents])


def download_gensim():
    import gensim.downloader as api

    # Model mic și corpus mic, utile pentru test offline
    resources = [
        "glove-wiki-gigaword-50",
        "text8",
    ]

    for res in resources:
        print(f"Downloading gensim resource: {res}")
        obj = api.load(res)
        print(f"Loaded gensim resource: {res} -> {type(obj)}")

    # Test rapid cu embeddings
    model = api.load("glove-wiki-gigaword-50")
    print("gensim vector size:", model.vector_size)
    print("gensim most similar to 'computer':", model.most_similar("computer", topn=5))


def download_transformers():
    from transformers import AutoTokenizer, AutoModel

    # Model mic, bun pentru warmup offline
    model_name = "distilbert-base-uncased"

    print(f"Downloading tokenizer/model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Salvare explicită locală, pe lângă cache
    local_dir = Path(os.environ["HF_HOME"]) / "local_models" / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)

    print(f"Saved local transformers model to: {local_dir}")

    # Test rapid
    import torch
    inputs = tokenizer("This is a test sentence.", return_tensors="pt")
    outputs = model(**inputs)
    print("Transformers hidden state shape:", tuple(outputs.last_hidden_state.shape))


def verify_tfidf_and_text_processing():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB

    docs = [
        "This is a simple document about machine learning.",
        "Natural language processing includes tokenization and tf idf.",
        "Transformers and embeddings are useful for NLP.",
        "This document is about text classification.",
    ]
    y = [0, 1, 1, 0]

    # stop_words="english" e builtin în scikit-learn; nu trebuie download separat
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vect.fit_transform(docs)

    print("TF-IDF shape:", X.shape)
    print("TF-IDF sample features:", vect.get_feature_names_out()[:20])

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", MultinomialNB()),
    ])
    pipe.fit(docs, y)
    pred = pipe.predict(["This text is about NLP and classification."])
    print("Pipeline prediction:", pred.tolist())


def verify_offline_load():
    # Verificare că modelul transformers poate fi încărcat din director local
    from transformers import AutoTokenizer, AutoModel

    local_dir = Path(os.environ["HF_HOME"]) / "local_models" / "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModel.from_pretrained(local_dir, local_files_only=True)

    inputs = tokenizer("Offline load works.", return_tensors="pt")
    outputs = model(**inputs)
    print("Offline local load OK:", tuple(outputs.last_hidden_state.shape))


def print_final_notes(base):
    print("\n" + "=" * 80)
    print("FINAL")
    print("=" * 80)
    print("Resursele au fost puse sub:")
    print(base)

    print("\nLa concurs poți seta variabilele astea înainte de rulare:")
    print(f"  export NLTK_DATA='{base / 'nltk_data'}'")
    print(f"  export HF_HOME='{base / 'hf_home'}'")
    print(f"  export GENSIM_DATA_DIR='{base / 'gensim_data'}'")
    print("  export TRANSFORMERS_OFFLINE=1")
    print("  export HF_HUB_OFFLINE=1")

    print("\nPe Windows (cmd):")
    print(f"  set NLTK_DATA={base / 'nltk_data'}")
    print(f"  set HF_HOME={base / 'hf_home'}")
    print(f"  set GENSIM_DATA_DIR={base / 'gensim_data'}")
    print("  set TRANSFORMERS_OFFLINE=1")
    print("  set HF_HUB_OFFLINE=1")


def main():
    base, _, _, _ = prepare_dirs()

    safe_step("Download NLTK resources", download_nltk)
    safe_step("Download spaCy English model", download_spacy)
    safe_step("Download gensim resources", download_gensim)
    safe_step("Download transformers model/tokenizer", download_transformers)
    safe_step("Verify TF-IDF and text processing", verify_tfidf_and_text_processing)
    safe_step("Verify offline transformers load", verify_offline_load)

    print_final_notes(base)


if __name__ == "__main__":
    main()
