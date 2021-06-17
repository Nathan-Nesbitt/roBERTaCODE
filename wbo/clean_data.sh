. ../venv/bin/activate
pip install gdown
gdown https://drive.google.com/uc?id=1f4y6oLg1745KvHMILYfLKHkQLvK3BWxh
unzip ETH\ Py150\ Open.zip
mv ETH\ Py150\ Open data
rm ETH\ Py150\ Open.zip
python clean_data.py