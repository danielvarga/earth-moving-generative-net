
# Attempt at faces:
# http://www.hcii-lab.net/data/SCUT-FBP/EN/introduce.html
cd ~/experiments/rbm/daniel-experiments/face/SCUT-FBP
wget http://www.hcii-lab.net/data/SCUT-FBP/download/Data_Collection.zip
wget http://www.hcii-lab.net/data/SCUT-FBP/download/Rating_Collection.zip
unzip -q Data_Collection.zip
cd Data_Collection
mkdir ../thumb
mogrify -path ../thumb -thumbnail 28x28 -extent 28x28 -gravity Center -colorspace gray *

