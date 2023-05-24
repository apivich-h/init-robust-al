conda env remove -n al-ntk; conda create -n al-ntk python=3.8
conda activate al-ntk
python -m pip install jax==0.3.13 jaxlib==0.3.10 neural-tangents==0.6.0
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install hydra-core==1.2.0 omegaconf==2.2.3 opt-einsum==3.3.0 Pillow==9.2.0 six==1.16.0 tqdm==4.63.2 baal==1.6.0 toma==1.1.0 