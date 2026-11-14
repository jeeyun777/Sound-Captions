# Read selected ESC-50 classes straight from the zip, resample 44.1k -> 16k, write int16 .raw
import sys, os, io, csv, zipfile, wave
import numpy as np
from scipy.signal import resample_poly

zpath, out = sys.argv[1], sys.argv[2]
KEEP = {
    'siren', 'car_horn', 'crying_baby',                              # events
    'engine', 'rain', 'wind', 'train', 'airplane', 'helicopter',     # backgrounds (car conditions)
    'dog', 'clock_alarm', 'church_bells', 'laughing', 'crow',        # distractors heard near a car (test only)
    'rooster', 'cat', 'door_wood_knock', 'glass_breaking', 'chainsaw', 'fireworks', 'sneezing', 'coughing',
    'vacuum_cleaner', 'keyboard_typing', 'footsteps', 'clapping', 'chirping_birds', 'thunderstorm', 'hand_saw', 'cow',  # other sounds (training only)
}
z = zipfile.ZipFile(zpath)
names = z.namelist()
root = names[0].split('/')[0]
meta = list(csv.DictReader(io.TextIOWrapper(z.open(f'{root}/meta/esc50.csv'), encoding='utf-8')))
n = 0
for row in meta:
    cat = row['category']
    if cat not in KEEP:
        continue
    fn = row['filename']
    if '/' in fn or '\\' in fn or '..' in fn:
        continue
    with wave.open(io.BytesIO(z.read(f'{root}/audio/{fn}'))) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, fn
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    if sr != 16000:
        x = resample_poly(x, 16000, sr)
    y = np.clip(np.round(x), -32768, 32767).astype('<i2')
    d = os.path.join(out, cat)
    os.makedirs(d, exist_ok=True)
    y.tofile(os.path.join(d, f"{row['fold']}_{fn[:-4]}.raw"))
    n += 1
print('clips', n)
for c in sorted(KEEP):
    print(c, len(os.listdir(os.path.join(out, c))))
