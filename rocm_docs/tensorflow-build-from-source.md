# hipTensorflow: Building From Source

## Intro
This instruction provides a starting point for a ROCm installation of hiptensorflow, build everything from source.
*Note*: it is recommended to start with a clean Ubuntu 16.04 system

## Install ROCm
```
export ROCM_PATH=/opt/rocm
export DEBIAN_FRONTEND noninteractive
sudo apt update && sudo apt install -y wget software-properties-common 
```

Add the ROCm repository:  
```
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
```
Install misc pkgs:
```
sudo apt-get update && sudo apt-get install -y \
  build-essential \
  clang-3.8 \
  clang-format-3.8 \
  clang-tidy-3.8 \
  cmake \
  cmake-qt-gui \
  ssh \
  curl \
  apt-utils \
  pkg-config \
  g++-multilib \
  git \
  libunwind-dev \
  libfftw3-dev \
  libelf-dev \
  libncurses5-dev \
  libpthread-stubs0-dev \
  vim \
  gfortran \
  libboost-program-options-dev \
  libssl-dev \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev \
  rpm \
  wget && \
  sudo apt-get clean && \
  sudo rm -rf /var/lib/apt/lists/*
```

Install ROCm pkgs:
```
sudo apt-get update && \
    sudo apt-get install -y --allow-unauthenticated \
    rocm-device-libs \
    hsa-ext-rocr-dev hsakmt-roct-dev hsa-rocr-dev \
    rocm-opencl rocm-opencl-dev \
    rocm-utils \
    rocm-profiler cxlactivitylogger && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*
```
Add some path variables:  
```
export HCC_HOME=$ROCM_PATH/hcc
export HIP_PATH=$ROCM_PATH/hip
export OPENCL_ROOT=$ROCM_PATH/opencl
export PATH="$HCC_HOME/bin:$HIP_PATH/bin:${PATH}"
export PATH="$ROCM_PATH/bin:${PATH}"
export PATH="$OPENCL_ROOT/bin:${PATH}"
```

Build HCC from source, and cleanup the default HCC to avoid issue:
```
sudo rm -rf /opt/rocm/hcc-1.0 && rm -rf /opt/rocm/lib/*.bc
cd ~ && git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git
cd ~ && mkdir -p build.hcc && cd build.hcc && cmake -DCMAKE_BUILD_TYPE=Release ../hcc
cd ~/build.hcc && make -j $(nproc) && make package && sudo dpkg -i hcc*.deb 
```

Build HIP from source:
```
cd ~ && git clone https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP.git
cd ~/HIP && mkdir -p build && cd build && cmake .. && make package -j && sudo dpkg -i *.deb 
```

Setup environment variables, and add those environment variables at the end of ~/.bashrc 
```
export HCC_HOME=/opt/rocm/hcc
export HIP_PATH=/opt/rocm/hip
export PATH=$HCC_HOME/bin:$HIP_PATH/bin:$PATH
```

## Install hipEigen
```
git clone -b develop https://github.com/ROCmSoftwarePlatform/hipeigen.git /opt/rocm/hipeigen
```

## Install MIOpen
```
cd ~ && git clone https://github.com/RadeonOpenCompute/rocm-cmake.git && cd ~/rocm-cmake && mkdir -p build && cd build && cmake .. && sudo cmake --build . --target install
cd ~ && git clone https://github.com/RadeonOpenCompute/clang-ocl.git && cd ~/clang-ocl && mkdir -p build && cd build && cmake .. && sudo cmake --build . --target install
cd ~ && git clone https://github.com/ROCmSoftwarePlatform/MIOpenGEMM.git && cd ~/MIOpenGEMM && mkdir -p build && cd build && cmake .. && sudo cmake --build . --target install
cd ~ && git clone https://github.com/ROCmSoftwarePlatform/MIOpen.git
cd ~/MIOpen && mkdir -p build && cd build && \ 
    CXX=/opt/rocm/hcc/bin/hcc cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/hcc;/opt/rocm/hip" -DCMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu/" .. -DMIOPEN_MAKE_BOOST_PUBLIC=ON && \
    sudo make install -j $(nproc)
```

## Install hcRNG and hcFFT 
```
git clone https://github.com/ROCmSoftwarePlatform/hcRNG.git ~/hcrng
cd ~/hcrng && bash build.sh && sudo dpkg -i ~/hcrng/build/*.deb

git clone https://github.com/ROCmSoftwarePlatform/hcFFT.git ~/hcfft
cd ~/hcfft && bash build.sh && sudo dpkg -i ~/hcfft/build/*.deb
```

## Install rocBLAS and hipBLAS 
```
cd ~ && git clone -b develop https://github.com/ROCmSoftwarePlatform/rocBLAS.git && cd rocBLAS && sudo ./install.sh -id 
cd ~ && git clone -b develop https://github.com/ROCmSoftwarePlatform/hipBLAS.git && cd hipBLAS && sudo ./install.sh -id 
```

## Install required python packages
```
sudo apt-get update && sudo apt-get install -y \
    python-numpy \
    python-dev \
    python-wheel \
    python-mock \
    python-future \
    python-pip \
    python-yaml \
    python-setuptools && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*
```

## Install bazel
```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y openjdk-8-jdk openjdk-8-jre unzip && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/* 
cd ~/ && wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-installer-linux-x86_64.sh 
sudo bash ~/bazel*.sh
```

## Build hiptf
```
# Clone it
cd ~ && git clone https://github.com/ROCmSoftwarePlatform/hiptensorflow.git

# Configure hiptf, need to enable HIP in configure options
cd ~/hiptensorflow && ./configure 

# Build and install hipTF pip package
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package &&
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg &&
pip uninstall tensorflow 
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.0.1-cp27-cp27mu-linux_x86_64.whl
```

## Clone hiptf models and benchmarks
```
cd ~ && git clone https://github.com/soumith/convnet-benchmarks.git
cd ~ && git clone https://github.com/tensorflow/models.git
```
