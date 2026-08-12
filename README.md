# Building-Damage-Assessment
an AI bulding damage assessment from a pre-disaster and a post-disaster images based on siam-unet and swin-unet 
## 🏗️ Technical Architecture

- **Damage Assessment:** Utilizes a **Siam-UNet** neural network to extract building damage levels from pre- and post-disaster remote sensing imagery.
- **Building Footprint Extraction:** Employs a **Swin-UNet** model to extract building contours from pre-disaster images, which are then automatically converted into vector shapefiles (`.shp`).
- **GUI Framework:** Built with **PyQt5**, integrating four core functional modules.

## 🚀 Core Features

| Module | Script | Description |
| :--- | :--- | :--- |
| **AI Inference** | `ai_infer.py` | Analyzes images to determine building damage levels and generates a damage raster TIF. |
| **Building Extraction** | `building_extract.py` | Extracts building footprints and exports them as vector shapefiles. |
| **RS Analysis** | `map_export.py` | Generates disaster classification maps (colored PNGs), raster TIFs, and disaster statistics. |
| **Report Generation** | `generate_pdf.py` | Consolidates the above outputs into a comprehensive disaster thematic map PDF. |

## 📦 Usage Instructions

This project comes with an **embedded Python environment**, making it extremely easy to get started:

1. Double-click `run.bat` to launch the application.
2. Select the pre-disaster and post-disaster TIF files for the same area.
3. Click the corresponding buttons to execute the desired functions.
4. All generated results will be automatically saved in the `output` folder.

## ⚠️ Note on Models

- The `.pth` model weights are stored in the `model` folder.
- Due to hardware limitations during the training phase, the current model accuracy may not be optimal. 
- **Improvement:** You can adjust the parameters in `model.py` and retrain the model for additional epochs until it fully converges to achieve better accuracy.

---

## 📊 Dataset

- **xBD Dataset** - 19 disaster types, 2,799 image pairs, 850,000+ building annotations
- **Damage Levels**: 0=No building, 1=No damage, 2=Minor, 3=Moderate, 4=Severe
- [Download xBD](https://xview2.org/dataset)

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB RAM recommended
- 8GB+ GPU memory recommended

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/SiamUNet-Damage-Assessment.git
cd SiamUNet-Damage-Assessment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download pretrained weights
# Download from [link] and place in ./model/
# - best_model_epoch55.pth (SiamUNet)
# - swinunet_best.pth (SwinUNet)

# 4. Configure paths
# Edit config.json to set your data paths

# 5. Run the application
python src/main.py
🚀 Quick Start
Using Conda (Alternative)
conda create -n damage_assessment python=3.10
conda activate damage_assessment
pip install -r requirements.txt
GUI Mode (Recommended)
python src/main.py
Command Line Mode
# AI inference only
python src/ai_infer.py --before pre.tif --after post.tif --out damage.tif

# Building extraction only
python src/building_extract.py --before pre.tif

# Full pipeline (damage + building + report)
python src/map_export.py --base post.tif --building building.shp --damage damage.tif --outtif overlay.tif
Batch Processing
python src/batch_processor.py --input_folder ./data/all_images --output_folder ./output/batch
📁 Project Structure
SiamUNet-Damage-Assessment/
├── src/
│   ├── ai_infer.py              # SiamUNet damage inference
│   ├── building_extract.py      # SwinUNet building extraction
│   ├── map_export.py            # GIS report generation
│   ├── generate_report_pdf.py   # PDF report generation
│   ├── batch_processor.py       # Batch processing
│   ├── model.py                 # Model definitions
│   └── main.py                  # PyQt5 GUI
├── model/                       # Pretrained weights
│   ├── best_model_epoch55.pth   # SiamUNet weights
│   └── swinunet_best.pth        # SwinUNet weights
├── output/                      # Results directory
├── template/                    # ArcGIS Pro templates
├── test/                        # Sample test data
├── requirements.txt
├── LICENSE
└── README.md
📊 Results
Performance Metrics
Metric	SiamUNet (Damage)	SwinUNet (Building)
Validation Loss	0.5004	-
Building IoU	-	0.79
Damage Classes	5	2 (binary)
Output Examples
Output Type	Description	Format
Damage Raster	Per-pixel damage classification (0-4)	GeoTIFF
Building Shapefile	Building footprints with damage attributes	SHP
Damage Overlay	RGB visualization of damaged buildings	GeoTIFF
Statistics	Damage counts and area by level	JSON/Excel
Report PDF	A3 landscape map with legend and stats	PDF
🗺️ GIS Output Details
Damage Overlay TIF
RGB raster with original image as base
Damaged buildings colored by severity:
Minor (Level 2): Yellow #ffcc00
Moderate (Level 3): Orange #ff8800
Severe (Level 4): Red #cc0000
Building Shapefile Attributes
Field	Type	Description
id	Integer	Unique building ID
damage_level	Integer	0-4 damage classification
area_m2	Float	Building area in square meters
px_cnt	Integer	Pixel count in raster
PDF Report
A3 Landscape (420mm × 297mm)
Damage map with legend
Statistics table
Damage area summary
🔧 Configuration
Edit config.json:

{
  "ai_env_python": "python",
  "arcgis_python": "path/to/arcgis/python.exe",
  "template_aprx_path": "./template/disaster/disaster.aprx",
  "output_folder": "./output",
  "gdb_name": "result.gdb"
}
📚 Citation
If you use this work in your research, please cite:

@misc{siamunet-damage-2024,
  author = {3401446793},
  title = {SiamUNet for Post-Disaster Building Damage Assessment},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/3401446793/SiamUNet-Damage-Assessment}
}
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgements
xBD Dataset - Building damage assessment dataset
SiamUNet - Change detection architecture
Google SKAI - Reference implementation
ArcGIS Pro - GIS integration
📧 Contact
Your Name - your.email@example.com

Project Link: https://github.com/yourusername/SiamUNet-Damage-Assessment

⭐ Star History

If you find this project useful, please give it a star! ⭐
