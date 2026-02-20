# OpenFace Docker image — builds FeatureExtraction binary
# Usage:
#   docker build -t being-openface -f docker/openface.Dockerfile .
#   docker run -v $(pwd)/data:/data being-openface \
#     FeatureExtraction -f /data/avatars/myavatar/myavatar.mp4 -out_dir /data/avatars/myavatar/
#
# Then rename the output CSV: mv data/avatars/myavatar/myavatar.csv data/avatars/myavatar/au.csv

FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libopenblas-dev liblapack-dev \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libatlas-base-dev gfortran \
    libjpeg-dev libpng-dev libtiff-dev \
    libboost-all-dev wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Build OpenCV 4.1 (OpenFace requirement)
WORKDIR /opt
RUN wget -q https://github.com/opencv/opencv/archive/4.1.0.zip -O opencv.zip && \
    unzip -q opencv.zip && rm opencv.zip && \
    mkdir opencv-4.1.0/build && cd opencv-4.1.0/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF .. && \
    make -j$(nproc) && make install

# Build dlib
RUN wget -q http://dlib.net/files/dlib-19.24.tar.bz2 && \
    tar xf dlib-19.24.tar.bz2 && rm dlib-19.24.tar.bz2 && \
    mkdir dlib-19.24/build && cd dlib-19.24/build && \
    cmake .. && make -j$(nproc) && make install

# Build OpenFace
RUN git clone https://github.com/TadasBaltrusaitis/OpenFace.git /opt/OpenFace && \
    cd /opt/OpenFace && \
    bash download_models.sh && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE .. && \
    make -j$(nproc)

# Runtime image
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libopenblas0 liblapack3 libgtk-3-0 \
    libavcodec58 libavformat58 libswscale5 \
    libjpeg8 libpng16-16 libtiff5 \
    libboost-filesystem1.74.0 libboost-system1.74.0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /opt/OpenFace/build/bin/ /usr/local/bin/
COPY --from=builder /opt/OpenFace/lib/ /opt/OpenFace/lib/
COPY --from=builder /opt/OpenFace/build/lib/ /opt/OpenFace/build/lib/

RUN ldconfig

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/FeatureExtraction"]
