from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HaMeR as a package',
    name='hamer',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy',
        'opencv-python==4.8.1.78',  # numpy 1.x 호환 (4.9+ 는 numpy>=2 요구)
        'pyrender',
        'pytorch-lightning==2.0.3',
        'scikit-image',
        'scikit-learn',
        'smplx==0.1.28',
        'tensorflow>=2.8.0',
        'torch',
        'torchvision',
        'yacs',
        # mmcv-full 1.x (prebuilt wheel)으로 별도 설치 - mmcv 2.x와 다른 패키지
        'timm==0.6.13',
        'einops',
        'xtcocotools',
        'pandas',
        'mediapipe',
    ],
    extras_require={
        'all': [
            'hydra-core',
            'hydra-submitit-launcher',
            'hydra-colorlog',
            'pyrootutils',
            'rich',
            'webdataset',
        ],
    },
)
