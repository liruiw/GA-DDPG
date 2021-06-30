# Download the pretrained model
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1erCIgqI2FvX-0B7ulg8qJs7eC2K73Smu' -O model.zip
echo "Model downloaded. Starting to unzip"
mkdir output
unzip -q model.zip -d .
rm model.zip

