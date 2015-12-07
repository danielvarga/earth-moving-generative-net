sudo apt-get --yes install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git > apt-get.cout 2> apt-get.cerr
# Check the latest deb at https://developer.nvidia.com/cuda-downloads
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb 
sudo apt-get update
sudo apt-get install cuda
sudo reboot
# check:
/usr/local/cuda-7.5/bin/nvcc --version

# This installs a Theano that's newer than regular pip install, actually this one:
# git+https://github.com/Theano/Theano.git@15c90dd3#egg=Theano==0.8.git
sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt

# if you have this repo available, copy or copypaste:
cp daniel-experiments/kohonen/theanorc.txt .theanorc

# check:
python -c "import numpy; numpy.test()"
python `python -c "import os, theano; print os.path.dirname(theano.__file__)"`/misc/check_blas.py

sudo pip install Lasagne==0.1

# Libs required for matplotlib that comes with nolearn.
# scikit-learn also comes with nolearn.
sudo apt-get install libpng-dev
sudo apt-get install libfreetype6-dev
sudo pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git
# otherwise matplotlib wants to communicate with nonexisting X11:
mkdir .matplotlib
echo "backend : Agg" > .matplotlib/matplotlibrc

mkdir .ssh
ssh-keygen -t rsa -b 4096 -C "daniel.varga@prezi.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
# Now add ~/.ssh/id_rsa.pub to github settings.
git config --global user.email "daniel.varga@prezi.com"
git config --global user.name "Daniel Varga"

git clone git@github.com:HIPS/Spearmint.git
sudo pip install -e Spearmint
sudo apt-get install mongodb
sudo pip install pymongo
sudo service mongod start

git clone git@github.com:danielvarga/daniel-experiments.git
# check:
time python daniel-experiments/kohonen/testNumpyToTheano.py > cout
# -> 9.5 secs for testSampleInitial(), 6.7 secs with allow_gc=False.
# test() minimal distances theano finishes in 0.263873 seconds.

wget http://deeplearning.net/data/mnist/mnist.pkl.gz
mv mnist.pkl.gz daniel-experiments/rbm/data/

cd daniel-experiments/kohonen
python Spearmint/spearmint/main.py . > spearmintOutput/log.cout 2> spearmintOutput/log.cerr
