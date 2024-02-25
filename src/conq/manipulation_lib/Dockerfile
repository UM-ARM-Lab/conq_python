FROM ubuntu:18.04

# Install packages without prompting the user to answer any questions
ENV DEBIAN_FRONTEND noninteractive 

# Update and install build dependencies
# Install initial dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    g++ \
    libeigen3-dev \
    libflann-dev \
    libboost-all-dev \
    libusb-1.0-0-dev \
    libusb-dev \
    libudev-dev \
    libfreenect-dev \
    libopenni2-dev \
    libopenni-dev \
    libvtk6-dev \
    libqhull-dev \
    libpcap-dev \
    wget \ 
    unzip \ 
    && rm -rf /var/lib/apt/lists/*

# Compile and install PCL 1.10
RUN git clone --branch pcl-1.10.0 https://github.com/PointCloudLibrary/pcl.git pcl-trunk \
    && cd pcl-trunk \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf pcl-trunk

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libvtk6-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and compile OpenCV 4.2
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip \
    && unzip opencv.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip \
    && unzip opencv_contrib.zip \
    && mkdir opencv-4.2.0/build && cd opencv-4.2.0/build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.2.0/modules \
      -D BUILD_EXAMPLES=ON .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd ../.. \
    && rm -rf opencv-4.2.0 \
    && rm -rf opencv_contrib-4.2.0 \
    && rm opencv.zip \
    && rm opencv_contrib.zip

# Clone and build the GPD repo
RUN git clone https://github.com/avazahedi/gpd \
    && cd gpd \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install

# Reset the DEBIAN_FRONTEND variable
ENV DEBIAN_FRONTEND=dialog

CMD ["bash"]

