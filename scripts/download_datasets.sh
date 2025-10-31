### For MVTecAD

echo "Downloading MVTecAD and MVTecLOCO datasets... It may take a while (about 10 miniutes). Coffee time!☕"

## make data directory if not exists
mkdir -p data/mvtec_ad
cd data/mvtec_ad
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xvf mvtec_anomaly_detection.tar.xz
rm mvtec_anomaly_detection.tar.xz

