# waifu_dataset_downloader.py
import json
import os
import requests
import urllib.parse
from pathlib import Path
import time
from tqdm import tqdm
import cv2

class WaifuDatasetDownloader:
    def __init__(self, output_folder="waifu_dataset"):
        self.output_folder = output_folder
        self.downloaded_count = 0
        self.failed_count = 0
        
    def download_from_json(self, json_path, max_downloads=None, resume=True):
        """
        Download dataset dari file JSON
        
        Args:
            json_path: Path ke file JSON metadata
            max_downloads: Maksimal file yang didownload (None = semua)
            resume: Lanjutkan download yang terputus
        """
        print(f"🔄 Starting download from {json_path}")
        
        # Buat folder output
        os.makedirs(self.output_folder, exist_ok=True)
        
        try:
            # Load JSON metadata
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse struktur JSON
            download_items = self._parse_json_structure(data)
            
            if not download_items:
                print("❌ No downloadable items found in JSON!")
                return False
            
            # Limit jumlah download jika diminta
            if max_downloads and max_downloads < len(download_items):
                download_items = download_items[:max_downloads]
            
            print(f"📥 Found {len(download_items)} items to download")
            
            # Download semua item
            self._download_all_items(download_items, resume)
            
            # Summary
            print(f"\n✅ Download completed!")
            print(f"   Successfully downloaded: {self.downloaded_count}")
            print(f"   Failed downloads: {self.failed_count}")
            print(f"   Output folder: {self.output_folder}")
            
            return True
            
        except FileNotFoundError:
            print(f"❌ File {json_path} not found!")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def _parse_json_structure(self, data):
        """Parse berbagai format struktur JSON"""
        items = []
        
        # Case 1: Array of objects
        if isinstance(data, list):
            for item in data:
                items.extend(self._extract_from_object(item))
        
        # Case 2: Object dengan berbagai struktur
        elif isinstance(data, dict):
            # Cari key yang berisi array gambar
            image_keys = ['images', 'data', 'items', 'waifus', 'characters', 'dataset']
            
            found_images = False
            for key in image_keys:
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        items.extend(self._extract_from_object(item))
                    found_images = True
                    break
            
            # Jika tidak ada array, coba extract langsung dari root object
            if not found_images:
                items.extend(self._extract_from_object(data))
        
        return items
    
    def _extract_from_object(self, obj):
        """Extract URL dan metadata dari object"""
        items = []
        
        if isinstance(obj, str):
            # Jika object adalah string URL langsung
            if self._is_valid_url(obj):
                items.append({
                    'url': obj,
                    'filename': self._get_filename_from_url(obj),
                    'metadata': {}
                })
        
        elif isinstance(obj, dict):
            # Cari field yang berisi URL
            url_fields = ['url', 'image_url', 'src', 'link', 'download_url', 'file_url', 'path']
            name_fields = ['name', 'filename', 'title', 'id', 'character_name', 'image_name']
            
            url = None
            filename = None
            
            # Cari URL
            for field in url_fields:
                if field in obj and self._is_valid_url(obj[field]):
                    url = obj[field]
                    break
            
            # Cari nama file
            for field in name_fields:
                if field in obj and obj[field]:
                    filename = str(obj[field])
                    break
            
            if url:
                if not filename:
                    filename = self._get_filename_from_url(url)
                else:
                    # Pastikan ada ekstensi
                    if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.png', '.jpeg', '.webp', '.gif']):
                        # Coba deteksi dari URL
                        url_filename = self._get_filename_from_url(url)
                        if '.' in url_filename:
                            ext = '.' + url_filename.split('.')[-1]
                        else:
                            ext = '.jpg'
                        filename += ext
                
                # Sanitize filename
                filename = self._sanitize_filename(filename)
                
                items.append({
                    'url': url,
                    'filename': filename,
                    'metadata': obj
                })
        
        return items
    
    def _is_valid_url(self, url):
        """Check if string is a valid URL"""
        if not isinstance(url, str):
            return False
        return url.startswith(('http://', 'https://'))
    
    def _get_filename_from_url(self, url):
        """Extract filename from URL"""
        parsed = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed.path)
        
        # Jika tidak ada filename yang jelas
        if not filename or '.' not in filename:
            # Generate filename dari hash URL
            filename = f"image_{abs(hash(url)) % 100000}"
            
            # Coba deteksi ekstensi dari URL atau default ke .jpg
            if any(ext in url.lower() for ext in ['.jpg', '.jpeg']):
                filename += '.jpg'
            elif '.png' in url.lower():
                filename += '.png'
            elif '.webp' in url.lower():
                filename += '.webp'
            elif '.gif' in url.lower():
                filename += '.gif'
            else:
                filename += '.jpg'
        
        return filename
    
    def _sanitize_filename(self, filename):
        """Clean filename dari karakter illegal"""
        illegal_chars = '<>:"/\\|?*'
        for char in illegal_chars:
            filename = filename.replace(char, '_')
        
        # Trim dan batasi panjang
        filename = filename.strip()[:200]
        
        return filename
    
    def _download_all_items(self, items, resume=True):
        """Download semua items dengan progress bar"""
        for item in tqdm(items, desc="Downloading images"):
            filepath = os.path.join(self.output_folder, item['filename'])
            
            # Skip jika file sudah ada dan resume=True
            if resume and os.path.exists(filepath):
                # Cek jika file valid
                if self._is_valid_image(filepath):
                    self.downloaded_count += 1
                    continue
                else:
                    # Hapus file corrupt
                    os.remove(filepath)
            
            # Download file
            success = self._download_single_file(item['url'], filepath)
            
            if success:
                self.downloaded_count += 1
            else:
                self.failed_count += 1
            
            # Delay kecil untuk menghindari rate limiting
            time.sleep(0.1)
    
    def _download_single_file(self, url, filepath):
        """Download single file"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Cek content type
            content_type = response.headers.get('content-type', '').lower()
            if 'image' not in content_type and not any(t in content_type for t in ['jpeg', 'png', 'jpg', 'webp']):
                print(f"⚠️  Non-image content: {url}")
                return False
            
            # Download
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verifikasi file
            if self._is_valid_image(filepath):
                return True
            else:
                os.remove(filepath)
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return False
    
    def _is_valid_image(self, filepath):
        """Check if downloaded file is a valid image"""
        try:
            img = cv2.imread(filepath)
            return img is not None and img.size > 0
        except:
            return False
    
    def create_sample_metadata(self, output_path="sample_waifu_metadata.json"):
        """Buat contoh file metadata JSON untuk testing"""
        
        # Format 1: Simple array of URLs
        sample_format_1 = [
            "https://example.com/waifu1.jpg",
            "https://example.com/waifu2.png",
            "https://example.com/waifu3.jpeg"
        ]
        
        # Format 2: Array of objects dengan metadata
        sample_format_2 = [
            {
                "id": "waifu_001",
                "name": "Anime Girl 1",
                "url": "https://example.com/characters/girl1.jpg",
                "character": "Sakura",
                "series": "Example Anime",
                "tags": ["pink_hair", "school_uniform"]
            },
            {
                "id": "waifu_002", 
                "name": "Anime Girl 2",
                "image_url": "https://example.com/characters/girl2.png",
                "character": "Misaka",
                "series": "Another Anime",
                "tags": ["brown_hair", "casual_clothes"]
            }
        ]
        
        # Format 3: Nested object structure
        sample_format_3 = {
            "dataset_info": {
                "name": "Waifu Collection",
                "version": "1.0",
                "total_count": 100
            },
            "images": [
                {
                    "filename": "character_001.jpg",
                    "download_url": "https://example.com/dataset/char001.jpg",
                    "metadata": {
                        "character_name": "Rei Ayanami",
                        "anime": "Evangelion",
                        "hair_color": "blue",
                        "eye_color": "red"
                    }
                },
                {
                    "filename": "character_002.png",
                    "src": "https://example.com/dataset/char002.png", 
                    "metadata": {
                        "character_name": "Asuka Langley",
                        "anime": "Evangelion",
                        "hair_color": "orange",
                        "eye_color": "blue"
                    }
                }
            ]
        }
        
        # Pilih format yang akan dibuat
        print("Select metadata format:")
        print("1. Simple URL array")
        print("2. Object array with metadata")
        print("3. Nested structure")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            sample_data = sample_format_1
        elif choice == "2":
            sample_data = sample_format_2
        else:
            sample_data = sample_format_3
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Sample metadata created: {output_path}")
        print("⚠️  Note: URLs in sample are examples only!")
        return output_path
    
    def validate_json(self, json_path):
        """Validasi file JSON dan tampilkan preview struktur"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ JSON file is valid: {json_path}")
            print(f"📊 Data type: {type(data).__name__}")
            
            if isinstance(data, list):
                print(f"📋 Array with {len(data)} items")
                if data:
                    print("🔍 First item preview:")
                    print(f"   Type: {type(data[0]).__name__}")
                    if isinstance(data[0], dict):
                        print(f"   Keys: {list(data[0].keys())}")
                    elif isinstance(data[0], str):
                        print(f"   Value: {data[0][:100]}...")
            
            elif isinstance(data, dict):
                print(f"📋 Object with keys: {list(data.keys())}")
                
                # Cari nested arrays
                for key, value in data.items():
                    if isinstance(value, list):
                        print(f"   📦 {key}: array with {len(value)} items")
                        if value and isinstance(value[0], dict):
                            print(f"      Keys in items: {list(value[0].keys())}")
            
            # Parse dan count URLs
            items = self._parse_json_structure(data)
            urls_found = len(items)
            
            print(f"🔗 Found {urls_found} downloadable URLs")
            
            if urls_found > 0:
                print("🎯 Sample URLs:")
                for i, item in enumerate(items[:3]):
                    print(f"   {i+1}. {item['filename']} <- {item['url'][:60]}...")
            
            return True, urls_found
            
        except FileNotFoundError:
            print(f"❌ File not found: {json_path}")
            return False, 0
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            return False, 0
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, 0
    
    def show_statistics(self):
        """Tampilkan statistik dataset yang sudah didownload"""
        if not os.path.exists(self.output_folder):
            print(f"❌ Output folder not found: {self.output_folder}")
            return
        
        # Hitung file gambar
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        image_files = []
        
        for ext in image_extensions:
            pattern = f"*{ext}"
            image_files.extend(Path(self.output_folder).glob(pattern))
            image_files.extend(Path(self.output_folder).glob(pattern.upper()))
        
        print(f"📊 Dataset Statistics")
        print(f"=" * 30)
        print(f"📁 Folder: {self.output_folder}")
        print(f"🖼️  Total images: {len(image_files)}")
        
        if image_files:
            # Analisis ukuran file
            total_size = sum(f.stat().st_size for f in image_files)
            avg_size = total_size / len(image_files)
            
            print(f"💾 Total size: {total_size / (1024*1024):.1f} MB")
            print(f"📏 Average size: {avg_size / 1024:.1f} KB")
            
            # Analisis ekstensi
            ext_count = {}
            for f in image_files:
                ext = f.suffix.lower()
                ext_count[ext] = ext_count.get(ext, 0) + 1
            
            print(f"📈 File types:")
            for ext, count in ext_count.items():
                print(f"   {ext}: {count} files")
            
            # Validasi beberapa file secara random
            import random
            sample_files = random.sample(image_files, min(10, len(image_files)))
            valid_count = sum(1 for f in sample_files if self._is_valid_image(str(f)))
            
            print(f"✅ Sample validation: {valid_count}/{len(sample_files)} files are valid")


def main():
    """Main function untuk command line usage"""
    print("🌟 Waifu Dataset Downloader")
    print("=" * 40)
    
    downloader = WaifuDatasetDownloader()
    
    while True:
        print("\nSelect an option:")
        print("1. Download from JSON metadata")
        print("2. Create sample metadata file") 
        print("3. Validate JSON file")
        print("4. Show dataset statistics")
        print("5. Change output folder")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            json_path = input("Enter JSON metadata path: ").strip()
            if not json_path:
                print("❌ Please provide a JSON file path!")
                continue
            
            # Validasi dulu
            valid, url_count = downloader.validate_json(json_path)
            if not valid or url_count == 0:
                continue
            
            # Opsi download
            max_downloads = input(f"Max downloads (Enter for all {url_count}): ").strip()
            max_downloads = int(max_downloads) if max_downloads.isdigit() else None
            
            resume = input("Resume previous download? (y/n, default=y): ").strip().lower()
            resume = resume != 'n'
            
            # Mulai download
            print(f"\n🚀 Starting download to: {downloader.output_folder}")
            success = downloader.download_from_json(json_path, max_downloads, resume)
            
            if success:
                # Tampilkan statistik
                downloader.show_statistics()
        
        elif choice == "2":
            output_path = input("Output path (default: sample_waifu_metadata.json): ").strip()
            if not output_path:
                output_path = "sample_waifu_metadata.json"
            downloader.create_sample_metadata(output_path)
        
        elif choice == "3":
            json_path = input("Enter JSON file path to validate: ").strip()
            if json_path:
                downloader.validate_json(json_path)
        
        elif choice == "4":
            downloader.show_statistics()
        
        elif choice == "5":
            new_folder = input(f"Current folder: {downloader.output_folder}\nNew folder: ").strip()
            if new_folder:
                downloader.output_folder = new_folder
                print(f"✅ Output folder changed to: {new_folder}")
        
        elif choice == "6":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()