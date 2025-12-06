import zipfile
import os

print("Extracting expert models...")
print("=" * 60)

models = [
    'td3-HalfCheetah-v3.zip',
    'td3-Hopper-v3.zip',
    'td3-Walker2d-v3.zip',
    'td3-Ant-v3.zip'
]

for model_file in models:
    zip_path = f'experts/{model_file}'
    extract_path = f'experts/{model_file.replace(".zip", "")}'

    print(f"\nExtracting {model_file}...")
    print(f"To: {extract_path}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('experts')
        print(f"[OK] Extracted successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to extract: {e}")

print("\n" + "=" * 60)
print("Extraction complete!")
print("\nListing extracted contents:")
os.system('dir /s /b experts\\*.pkl')
os.system('dir /s /b experts\\*.zip')
