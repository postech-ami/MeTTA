from setuptools import setup, find_packages

setup(
    name                = 'fantasia3d',
    version             = '0.0.1',
    description         = 'fantasia3d text-to-3d, image-to-3d package',
    author              = 'ugkim',
    author_email        = 'ugkim@postech.ac.kr',
    url                 = 'https://github.com/ug-kim/fantasia3d',
    install_requires    =  ["rich",
                            "tqdm",
                            "ninja",
                            "numpy" ,
                            "scipy",
                            "trimesh",
                            "torch-ema",
                            "matplotlib",
                            "tensorboardX",
                            "opencv-python",
                            "imageio",
                            "imageio-ffmpeg",
                            "scikit-learn",
                            "xatlas",
                            "PyOpenGL",
                            "glfw",
                            "accelerate",
                            "huggingface_hub",
                            "diffusers>=0.9.0", # for stable-diffusion 2.0
                            "transformers",
                            "open3d",
                            # "git+https://github.com/NVlabs/nvdiffrast/",
                            ],
    packages            = find_packages(exclude = []),
    keywords            = ['fantasia3d'],
    python_requires     = '>=3.7',
    package_data        = {},
    zip_safe            = False,
)