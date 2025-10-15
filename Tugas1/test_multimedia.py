import importlib

libraries = [
    ("librosa", "Audio"),
    ("soundfile", "Audio"),
    ("scipy", "Audio"),
    ("cv2", "Image"),
    ("PIL", "Image"),
    ("skimage", "Image"),
    ("matplotlib", "Image"),
    ("moviepy", "Video"),
    ("numpy", "General"),
    ("pandas", "General"),
    ("jupyter", "General"),
]

print("Cek instalasi library multimedia:\n")

for lib, category in libraries:
    try:
        module = importlib.import_module(lib)
        version = getattr(module, "__version__", "versi tidak diketahui")
        print(f"[SUKSES] {lib} ({category}) terinstall, versi: {version}")
    except Exception as e:
        print(f"[GAGAL]  {lib} ({category}) TIDAK terinstall. ({e})")
