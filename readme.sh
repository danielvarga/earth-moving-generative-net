
# Attempt at faces:
# http://www.hcii-lab.net/data/SCUT-FBP/EN/introduce.html
cd ~/experiments/rbm/daniel-experiments/face/SCUT-FBP
wget http://www.hcii-lab.net/data/SCUT-FBP/download/Data_Collection.zip
wget http://www.hcii-lab.net/data/SCUT-FBP/download/Rating_Collection.zip
unzip -q Data_Collection.zip
cd Data_Collection
mkdir ../thumb
mogrify -path ../thumb -thumbnail 28x28 -extent 28x28 -gravity Center -colorspace gray *.jpg

cd ~/experiments/rbm/daniel-experiments/kohonen
montage ../face/SCUT-FBP/thumb/SCUT-FBP-*[0-7].jpg -geometry 28x28+0+0 ../face/SCUT-FBP/tile.jpg

mkdir ~/experiments/rbm/daniel-experiments/face/SCUT-FBP/thumb.big
mogrify -path ~/experiments/rbm/daniel-experiments/face/SCUT-FBP/thumb.big -thumbnail 32x42 -extent 32x42 -gravity Center -colorspace gray -format png ~/experiments/rbm/daniel-experiments/face/SCUT-FBP/Data_Collection/*.jpg
# -> this became exp.bigfaces.n100 after learning.

# Second attempt at faces:
# https://www.kaggle.com/c/facial-keypoints-detection/data
# http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#the-data
cd ~/experiments/rbm/daniel-experiments/face/kaggle-facial-keypoints-detection
( cat training.csv | awk 'BEGIN{FS=","} { print $NF }' | tail -n +2 ; cat test.csv | cut -f2 -d',' | tail -n +2 ) > pixels.txt
# -> 7049 train + 1784 test = 8832 96x96x1 images.
