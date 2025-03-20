# Position-Aware Self-Attention with Momentum Contrast for Semantic-Augmented Cross-Modal Retrieval(PSMC)
# Requirements and Installation
We recommended the following dependencies.
  - Python 3.7
  - Pytorch 1.6+
  - Numpy
Our source code of PSMC accepted by TIP will be released as soon as possible. It is built on top of the [vse_inf](https://github.com/woodfrog/vse_infty) in PyTorch. 
# Download data 
```
data
├── f30k
│   ├── precomp  # pre-computed BUTD region features for Flickr30K, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── flickr30k-images   # raw coco images
│   │      ├── xxx.jpg
│   │      └── ......
│   └── id_mapping.json  # mapping from f30k index to image's file name
│
├── coco
│   ├── precomp  # pre-computed BUTD region features for COCO, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── images   # raw coco images
│   │      ├── train2014
│   │      └── val2014
│   │
│   └── id_mapping.json  # mapping from coco-id to image's file name
│
└── vocab  # vocab files provided by SCAN (only used when the text backbone is BiGRU)
```
The download links for original COCO/F30K images, precomputed BUTD features, and corresponding vocabularies are from the official repo of [SCAN](https://github.com/kuanghuei/SCAN#download-data). The ```precomp``` folders contain pre-computed BUTD region features, ```data/coco/images``` contains raw MS-COCO images, and ```data/f30k/flickr30k-images``` contains raw Flickr30K images. 
Because the download link for the pre-computed features in [SCAN]((https://github.com/kuanghuei/SCAN)) is seemingly taken down. The [link](https://www.dropbox.com/sh/qp3fw9hqegpm914/AAC3D3kqkh5i4cgZOfVmlWCDa?dl=0) provided by the author of [vse_infty](https://github.com/woodfrog/vse_infty) contains a copy of these files. 


## Training
Train Flickr30K and MSCOCO fromm scratch:- Train new PSMC models: Run `train.py`:
```bash
python train.py --data_path "$DATA_PATH" --data_name "$DATA_NAME" --logger_name "$LOGGER_NAME" 
```

## Evaluation
Modify the corresponding arguments in `eval.py` and run `python eval.py`.


Please use the following bib entry to cite this paper if you are using any resources from the repo.
```
@article{zhu2025psmc,
  title={Position-Aware Self-Attention with Momentum Contrast for Semantic-Augmented Cross-Modal Retrieval},
  author={Zhu, },
  journal={IEEE Transactions on Image Processing},
  year={2025},
  publisher={IEEE}
}
```
