ARG BASE_IMAGE=docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1
FROM ${BASE_IMAGE}

USER root

ARG VGGT_COMMIT=a288dd0f14786c93483e45524328726ab7b1b4ce

COPY geometry_teacher/deploy/.downloads/vggt-${VGGT_COMMIT}.tar.gz /tmp/vggt.tar.gz

RUN python -m pip uninstall --yes bpy rembg opencv-python opencv-python-headless \
    && python -m pip install --no-cache-dir \
        'numpy<2' \
        Pillow \
        einops \
        huggingface_hub \
        'opencv-python==4.11.0.86' \
        safetensors \
        'transformers==5.14.1' \
    && mkdir -p /opt/official/vggt \
    && tar --extract --gzip --file /tmp/vggt.tar.gz \
        --directory /opt/official/vggt --strip-components=1 \
    && python -m pip install --no-cache-dir --no-deps -e /opt/official/vggt \
    && printf '%s\n' "${VGGT_COMMIT}" > /opt/official/VGGT_COMMIT \
    && rm /tmp/vggt.tar.gz \
    && python -m pip check \
    && python -c 'import cv2; from transformers import AutoImageProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor, VitPoseForPoseEstimation; from vggt.models.vggt import VGGT; from vggt.utils.geometry import depth_to_cam_coords_points; from vggt.utils.pose_enc import pose_encoding_to_extri_intri'

ENV PYTHONPATH=/workspace/project \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/project

CMD ["bash"]
