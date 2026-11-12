import glob
import os
import random
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from app.services.image_audio.config import settings
from app.services.image_audio.audio import decode_audio, make_windows
from app.services.image_audio.detectors.acoustic_forensics import extract_features_from_samples

print("Extracting features using production decode_audio and make_windows pipeline...", flush=True)

categories_fake = {
    'VoiceBox': 104,
    'xTTS': 200,
    'FlashSpeech': 118,
    'seedtts_files': 150,
    'VALLE': 95,
    'NaturalSpeech3': 32,
    'PromptTTS2': 25,
}

X = []
y = []
src_names = []

for cat, file_limit in categories_fake.items():
    files = glob.glob(rf'C:\Users\Faraaz Khatib\Downloads\VeriTrust-main\VeriTrust-main\{cat}\*.wav')[:file_limit]
    cat_wins = 0
    for f in files:
        try:
            with open(f, 'rb') as fp:
                decoded = decode_audio(fp.read(), settings)
            windows, _ = make_windows(decoded, settings)
        except Exception:
            continue
        for w in windows[:4]:
            if not w.silent:
                feats = extract_features_from_samples(w.samples, decoded.sample_rate)
                if feats is not None:
                    X.append(feats)
                    y.append(1) # Fake
                    src_names.append(cat)
                    cat_wins += 1
    print(f"  {cat:<15}: {len(files)} files -> {cat_wins} windows", flush=True)

real_files = glob.glob(r'C:\Users\Faraaz Khatib\Downloads\VeriTrust-main\VeriTrust-main\real_samples\*.wav')[:650]
real_wins = 0
for f in real_files:
    try:
        with open(f, 'rb') as fp:
            decoded = decode_audio(fp.read(), settings)
        windows, _ = make_windows(decoded, settings)
    except Exception:
        continue
    for w in windows[:3]:
        if not w.silent:
            feats = extract_features_from_samples(w.samples, decoded.sample_rate)
            if feats is not None:
                X.append(feats)
                y.append(0) # Real
                src_names.append('real')
                real_wins += 1
print(f"  {'real_samples':<15}: {len(real_files)} files -> {real_wins} windows", flush=True)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
src = np.array(src_names)

print(f"\nTotal training windows: {len(X)} (Fake: {np.sum(y==1)}, Real: {np.sum(y==0)})", flush=True)

# Train/Test Split
X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
    X, y, src, test_size=0.20, random_state=42, stratify=y
)

clf = HistGradientBoostingClassifier(
    random_state=42,
    max_iter=250,
    max_leaf_nodes=35,
    min_samples_leaf=15,
    learning_rate=0.08
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\n=== Window-Level Held-out Test Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

print("=== Per-Generator Test Accuracy on 4s Windows ===")
for gen in np.unique(src_test):
    idx = (src_test == gen)
    gen_acc = accuracy_score(y_test[idx], y_pred[idx])
    gen_prob = np.mean(y_prob[idx])
    print(f"{gen:<15} ({np.sum(idx):<4} windows): Accuracy = {gen_acc*100:.1f}%, Mean Fake Prob = {gen_prob:.3f}")

# Train on full dataset and save
print("\nRefitting on all available data for production weights...", flush=True)
clf_full = HistGradientBoostingClassifier(
    random_state=42,
    max_iter=300,
    max_leaf_nodes=35,
    min_samples_leaf=15,
    learning_rate=0.08
)
clf_full.fit(X, y)

weights_dir = r"c:\Users\Faraaz Khatib\Downloads\VeriTrustAI-mainmain\VeriTrustAI-main\backend\app\services\image_audio\weights"
os.makedirs(weights_dir, exist_ok=True)
save_path = os.path.join(weights_dir, "acoustic_forensics.joblib")
joblib.dump(clf_full, save_path)
print(f"Saved calibrated model to: {save_path} (size: {os.path.getsize(save_path) / 1024:.1f} KB)", flush=True)
