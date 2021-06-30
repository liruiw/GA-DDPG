# Download the data
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1EdFRu2GK3p6clMAXT0fM_wPePE0AWMEc' -O data.zip
echo "Data downloaded. Starting to unzip"
unzip  data.zip  -d data
rm data.zip
