mkdir -p data/LLaVA-Instruct
mkdir -p data/LLaVA-Pretrain

wget -O data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json?download=true"
wget -O data/LLaVA-Instruct/llava_v1_5_mix665k.json "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json?download=true"

wget -O data/LLaVA-Pretrain/images.zip "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true"

mkdir -p data/LLaVA-Instruct/coco
mkdir -p data/LLaVA-Instruct/gqa
mkdir -p data/LLaVA-Instruct/ocr_vqa
mkdir -p data/LLaVA-Instruct/textvqa
mkdir -p data/LLaVA-Instruct/vg

wget -O data/LLaVA-Instruct/coco/train2017.zip http://images.cocodataset.org/zips/train2017.zip
wget -O data/LLaVA-Instruct/gqa/images.zip https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
wget -O data/LLaVA-Instruct/ocr_vqa/images.tar https://huggingface.co/datasets/ej2/llava-ocr-vqa/resolve/main/ocr_vqa.tar?download=true
wget -O data/LLaVA-Instruct/textvqa/train_images.zip https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
wget -O data/LLaVA-Instruct/vg/VG_100K.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget -O data/LLaVA-Instruct/vg/VG_100K_2.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

unzip data/LLaVA-Pretrain/images.zip -d data/LLaVA-Pretrain/images

unzip data/LLaVA-Instruct/coco/train2017.zip -d data/LLaVA-Instruct/coco
unzip data/LLaVA-Instruct/gqa/images.zip -d data/LLaVA-Instruct/gqa
tar -xvf data/LLaVA-Instruct/ocr_vqa/images.tar -C data/LLaVA-Instruct/ocr_vqa
unzip data/LLaVA-Instruct/textvqa/train_images.zip -d data/LLaVA-Instruct/textvqa
unzip data/LLaVA-Instruct/vg/VG_100K.zip -d data/LLaVA-Instruct/vg
unzip data/LLaVA-Instruct/vg/VG_100K_2.zip -d data/LLaVA-Instruct/vg
