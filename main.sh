tar -zxvf info.tar.gz -C ./data
cd model1
tar -zxvf ckpt.tar.gz
python validation.py
cd ../model2
tar -zxvf ckpt.tar.gz
python validation.py
cd ..
python ensemble.py

