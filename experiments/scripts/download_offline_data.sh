# Download the data
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1ZhmVq3J82t5mByGKngSliknNZZhiLzxg' -O data.zip
echo "Data downloaded. Starting to unzip"
unzip data.zip  -d data/offline_data/
rm data.zip
