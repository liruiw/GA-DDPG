# Download the data
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=136rLjyjFFRMyVxUZT6txB5XR2Ct_LNWC' -O data.zip
echo "Data downloaded. Starting to unzip"
unzip  data.zip  -d data
rm data.zip
