. /Users/daniel/experiments/rbm/daniel-experiments/lasagne-demo/venv/bin/activate

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

cd exp.bigfaces.n100
ssh kruso.mokk.bme.hu mkdir ./public_html/kohonen
for dir in xy yz xz s ; do convert $dir[1-9]0000.png $dir[0-9][0-9]0000.png -delay 10 -loop 0 $dir.gif ; done
scp *.gif kruso.mokk.bme.hu:./public_html/kohonen/


# Second attempt at faces:
# https://www.kaggle.com/c/facial-keypoints-detection/data
# http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#the-data
cd ~/experiments/rbm/daniel-experiments/face/kaggle-facial-keypoints-detection
( cat training.csv | awk 'BEGIN{FS=","} { print $NF }' | tail -n +2 ; cat test.csv | cut -f2 -d',' | tail -n +2 ) > pixels.txt
# -> 7049 train + 1784 test = 8832 96x96x1 images.


# Back at digits again, playing with larger n:
# exp.20dCubeMixture.2layerTanh.n100.digit7 is a stupid mistake. Both
python generative-mlp.py exp.20dCubeMixture.2layerTanh.n100.digit7 100 > cout.exp.20dCubeMixture.2layerTanh.n100.digit7
# and
python generative-mlp.py exp.20dCubeMixture.2layerTanh.n100.digit7 300 > cout.exp.20dCubeMixture.2layerTanh.n300.digit7
# were pointing to this dir. The n300 started later, so it's overwritten the other, except for
# the latest images, *101700.png - *102800.png. n300 is definitely worse,
# more prone to forked lines. Why?

# Now running:
python generative-mlp.py exp.20dCubeMixture.2layerTanh.n100.digit3 100 > cout.exp.20dCubeMixture.2layerTanh.n100.digit3
python generative-mlp.py exp.20dGaussian.2layerTanh.n300.digit3 300 > cout.exp.20dGaussian.2layerTanh.n300.digit3
# UPDATE: Dumb me, that's n300 right there.
# Have to do Gaussian again with n100. See below.

# The filenames tell all, hopefully. For the record:
# 100 hidden units, learning_rate=0.02, momentum=0.5
# scale of normal distribution 1/4, findGenForData True, overSamplingFactor 1.

# UPDATE: Disregard this paragraph, it compares gauss.n300 to mixture.n100.
# -> After some 2000 epochs, the main difference is that mixture does the forks,
# gauss doesn't, but gauss is super non-diverse.
# After some 10000-30000 epochs (pretty subjective when) mixture stops doing the forks.
# The weirdest is that around here, gauss starts the fork thing, while still not
# being as diverse as mixture. All in all, it's objectively worse.

# UPDATE: Apples to apples aka n100 to n100 comparison
# between mixture and gauss.
# and also between gauss.n300 and gauss.n100.
python generative-mlp.py exp.20dGaussian.2layerTanh.n100.digit3 100 > cout.exp.20dGaussian.2layerTanh.n100.digit3
# -> EVALUATE!

# Okay, let's go all in, how about getting rid of the continuous component?
python generative-mlp.py exp.20dBoolean.2layerTanh.n100.digit3 100 > cout.exp.20dBoolean.2layerTanh.n100.digit3
python generative-mlp.py exp.50dBoolean.2layerTanh.n100.digit3 100 > cout.exp.50dBoolean.2layerTanh.n100.digit3
# -> EVALUATE!

# Does this fully-boolean-craziness work with more diverse data as well?
python generative-mlp.py exp.50dBoolean.2layerTanh.n100.digitAll 100 > cout.exp.50dBoolean.2layerTanh.n100.digitAll
# -> EVALUATE!

python generative-mlp.py exp.20dCubeMixture.2layerTanh.n100.digit2 100 > cout.exp.20dCubeMixture.2layerTanh.n100.digit2
python generative-mlp.py exp.50dCubeMixture.2layerTanh.n100.digit2 100 > cout.exp.50dCubeMixture.2layerTanh.n100.digit2
# -> Waiting for results. EVALUATE!

# Lots of work done on quantifying generation performance.
# We sample train and validation (unseen), greedily approximate them with generated samples,
# and quantify/visualize difference. Specifically, we log total L2 diff on train and valid,
# we visualize difference, and histogram L2 distances between sample and best surrogate.
python generative-mlp.py exp.20dCubeMixture.2layerTanh.n100.digit2.moreVis 100 > cout.exp.20dCubeMixture.2layerTanh.n100.digit2.moreVis

