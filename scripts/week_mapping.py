"""W2-W5 練習トラック — どの問題をどの週に割り当てるか。

`build_weeks.py` がこれを読んで `practice/W{n}/` を生成する。
順番は **学習順**（最初に解くべきものから）で書く。"""

WEEK_MAPPING = {
    "W2": {
        "title": "MLP / 基本分類 / 基礎 optimization",
        "intro": (
            "MLP の forward → activation → loss → optimizer → 評価指標 までの一周。"
            "基本的な MNIST / 分類パイプラインの構成部品を一つずつ自前実装する週。"
        ),
        # 学習順：simplest model → layer → activation → loss → optimizer → eval
        "problems": [
            "40_linear_regression",
            "03_linear",
            "01_relu",
            "02_softmax",
            "16_cross_entropy",
            "53_nll_loss",
            "54_sgd_momentum",
            "52_top_k_accuracy",
        ],
    },
    "W3": {
        "title": "正則化 / 正規化 / advanced optimization",
        "intro": (
            "深いネットを安定して訓練するための道具一式。"
            "init → normalization → regularization → optimizer → schedule → stability の順で学習。"
        ),
        # init → norm → regularize → optimize → schedule → stability
        "problems": [
            "20_weight_init",
            "07_batchnorm",
            "04_layernorm",
            "17_dropout",
            "55_weight_decay",
            "29_adam",
            "56_adamw",
            "30_cosine_lr",
            "21_gradient_clipping",
        ],
    },
    "W4": {
        "title": "CNN 基礎 + 基本 transforms",
        "intro": (
            "CIFAR-10 CNN を構成する全部品。Conv → Pool → GAP の forward + "
            "基本 data augmentation (Normalize / HFlip / RandomCrop) のセット。"
        ),
        # conv → pooling 各種 → transforms
        "problems": [
            "22_conv2d",
            "41_maxpool2d",
            "49_avg_pool2d",
            "50_global_avg_pool",
            "42_normalize",
            "43_random_hflip",
            "44_random_crop",
        ],
    },
    "W5": {
        "title": "CIFAR-10 advanced レシピ",
        "intro": (
            "CIFAR-10 で 95% → 97% に押し上げる「効くやつ」5点。"
            "Label smoothing + 強い augmentation (Cutout / Mixup / CutMix) + 推論時 TTA。"
        ),
        # label smoothing → cutout (single) → mixup → cutmix → TTA
        "problems": [
            "51_label_smoothing_ce",
            "45_cutout",
            "46_mixup",
            "47_cutmix",
            "48_tta_hflip",
        ],
    },
}
