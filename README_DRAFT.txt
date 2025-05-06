Facebpok SAM2 setup: http://github.com/facebookresearch/sam2

https://github.com/facebookresearch/sam2/blob/main/INSTALL.md

https://github.com/facebookresearch/sam2/issues/22

pip install --upgrade huggingface_hub

COLMAP + PyCOLMAP with CUDA: https://colmap.github.io/pycolmap/index.html (Linux)

Under Ubuntu 22.04, there is a problem when compiling with Ubuntu’s default CUDA package and GCC, and you must compile against GCC 10:

sudo apt-get install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
# ... and then run CMake against COLMAP's sources.

Conda env need to be disabled temporarily for COLMAP

Triposr install
https://github.com/VAST-AI-Research/TripoSR
pip install onnxruntime-gpu